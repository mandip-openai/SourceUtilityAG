import asyncio
import random
import typing

import pydantic
import rich

import chz
import maraschino_rater.utils.lm_utils as lm_utils
import sonic_training.experiments.index_switcher.v2.grader.utils as grader_utils
from bus_token_completer import BusTokenCompleter
from chat import chat
from chat.render.v4.experimental.strawberry.formatter import BerryChannel
from maraschino_rater.base import Document, Query
from message_completer.berry_selective_message_completer import BerrySelectiveMessageCompleter
from oai_serialization import oai_json
from search_service.api.user_metadata import UserMetadata
from sonic_training.experiments.tools.lean_browser_sampling import load_completer, make_convo

"""
brix git pull && \
oaipkg run maraschino_data_pipeline.grader_experiment.grade \
  rating_mode=snippet_wo_context \
  model=neutrino \
  grading_concurrency=30 \
  input_path=az://oairic1/oaimaraschino/datasets/source_utility/experiments/ds=20250123/identity=685f9d0e-1b03-442e-b09d-b3339e2a4340/content_enriched_search_result_groups_dump.jsonl \
  output_path=az://oairic1/oaibwen/data/source_utility/experiments/ds=20250123/identity=685f9d0e-1b03-442e-b09d-b3339e2a4340/content_enriched_search_result_groups_dump.jsonl \
  random_seed=42 \
  topic=bwen-citation-neutrino-grader-0119 \
  limit=30
"""

TModelType = typing.Literal[
    "tb2_with_browse", "cb", "neutrino", "gpt-4o-mini", "gpt4o-sonic-training", "nv4"
]
TRatingMode = typing.Literal["page", "snippet_w_context", "snippet_wo_context"]


def none_throws(x: typing.Any) -> typing.Any:
    assert x is not None
    return x


class DocumentWithContext(pydantic.BaseModel):
    document: Document
    context: str | None = None


DEFAULT_INSTRUCTION = """
You are a professional grader that judges the quality of search results.
You need to follow the rating instruction below to rate a given QUERY - DOCUMENT pair with respect to each dimension.
The QUERY contains the QUERY TEXT, DATE, and LOCATION of the searcher. The DOCUMENT contains TITLE, URL, PUBLICATION DATE, and CONTENT.
Optionally, the DOCUMENT may also be provided with an extra CONTEXT, which contains the text of the page that the CONTENT is extracted from.
When it is provided, you can refer to it to make your decision. But the grading should focus on CONTENT. If the key information is missing from CONTENT but in CONTEXT, you rate it as low.
Write your decision for each aspects in the markdown format. It should contain a title of the area, the rating, and detailed justification of your decision.
If you have any confusion or something worth commenting, can also output them in a "notes".

The response of each aspect (dimension) is:
## <Aspect>
* rating: <rating>
* justification: <justification>
* notes: <notes>

The Rubrics is:

"""

DEFAULT_RUBRICS = r"""
To evaluate search results effectively, think like a professional rater using a structured approach. Your goal is to assess each result’s quality based on its ability to meet user needs, focusing on the following key factors that you are asked below on Accuracy of the content. Follow these instructions:

1. Step into the user’s shoes: Consider the user’s intent behind the query. What specific information or action might the user need? Tailor your evaluation accordingly.
2. Use the guidelines as a framework: Evaluate based on nuanced distinctions. Avoid broad generalizations—consider the context, specificity, and depth of the result.
3. Identify strengths and weaknesses: For each evaluation, pinpoint what the result does well and where it falls short, using examples from the content.
4. Assign a rating on a 1-7 scale for each category, with 1 representing the lowest and 7 the highest level of factor you are evaluating. Justify your rating with concrete reasoning that aligns with the established evaluation criteria. Be concise in your justification with less than 20 words.
5. Apply Evaluation Levels guidance to provide a precise rating.

#### 4\. Accuracy Evaluation

Prompt:

"Analyze the factual accuracy of the content. Adopt the mindset of a rater who is cross-referencing trusted sources and questioning unsupported claims. Evaluate the following

Evaluation Criteria:

Are the facts correct and well-supported?
    * Look for evidence or citations supporting the claims. Flag any misinformation.
    * Example of high accuracy: A result for 'symptoms of COVID-19' citing CDC or WHO guidelines.
    * Example of low accuracy: A blog post with anecdotal symptoms contradicting medical science.
Does the result omit or distort information?
    * Check for completeness and whether the result presents balanced perspectives.
    * Example of high completeness: An article covering both the pros and cons of renewable energy.
    * Example of low completeness: A biased article discussing only the negatives of renewable energy.

Instruction: Provide a rating (High, Medium, or Low) with a justification based on these factors. Imagine you are fact-checking as a professional editor. Verify claims, identify unsupported assertions, and consider whether the content could mislead users.

Evaluation Levels: Unless specified otherwise, use the following scale to classify evaluation levels: 
    * Low (1-2): Contains factual inaccuracies or unsupported claims.
    * Medium (3-5):  Generally accurate but lacks supporting evidence or has minor errors.
    * High (6-7): Fully accurate, with claims supported by credible sources.

Please interpolate within the designated range as appropriate to reflect nuances in the evaluation."
"""

class Grader:
    def __init__(self, instruction: str = DEFAULT_INSTRUCTION, debug: bool = False):
        self.instruction = instruction
        self.debug = debug

    async def __call__(
        self, query: Query, document: Document | DocumentWithContext, rubrics: str = ""
    ):
        date_spec = ""
        if query.create_timestamp:
            date_spec = f"{query.create_timestamp.date()}"
        geo_spec = ""
        if (
            query.user_metadata
            and query.user_metadata.user_country
            and query.user_metadata.user_region
            and query.user_metadata.ip_city
        ):
            geo_spec = f"country - {query.user_metadata.user_country}, region - {query.user_metadata.user_region}, city - {query.user_metadata.ip_city}"

        context = None
        doc = None
        if isinstance(document, DocumentWithContext):
            context = document.context
            doc = document.document
        else:
            doc = document

        input = f"""
# Input Query and Document

QUERY:
QUERY TEXT: {query.query}
DATE: {date_spec or "N/A"}
LOCATION: {geo_spec or "N/A"}

DOCUMENT:
TITLE: {doc.title or "N/A"}
URL: {doc.url}
PUBLICATION DATE: {doc.pub_date or "N/A"}
CONTEXT: {context or "N/A"}
CONTENT: {doc.content or "N/A"}
    """
        try:
            return await self.rate_text(input, rubrics)
        except Exception as e:
            return f"Error: {e}"

    async def rate_text(self, text: str, rubrics: str):
        raise NotImplementedError()


class ToolberryWithBrowseGrader(Grader):
    async def rate_text(self, text: str, rubrics: str):
        convo = make_convo(
            "\n".join(
                [
                    self.instruction,
                    rubrics,
                    text,
                    "You can use browser.tool_call to help you grading. ",
                ]
            )
        )
        if self.debug:
            print(f"DEBUG: {convo=}")

        tc = await load_completer()
        result = None
        try:
            async for result in tc.async_completion_stream(convo, include_system_messages=False):
                pass
            assert result is not None
            if self.debug:
                for message in result.input_conversation.messages:
                    print(f"input: {message=}")
                for message in result.output_messages:
                    print(f"output: {message=}")
            return result.output_messages[-1].content.model_dump_json()
        except Exception as e:
            return f"Error: {e}"


class ChatberryGrader(Grader):
    def __init__(self, juice=128, instruction=DEFAULT_INSTRUCTION, debug=False):
        super().__init__(instruction, debug)
        self.juice = juice

    async def rate_text(self, text: str, rubrics: str):
        r, _cot = await grader_utils.query_chatberry_parsed(
            prompt="\n".join([self.instruction, rubrics, text]), reward_multipler=self.juice
        )
        return r


class NeutrinoV4Grader(Grader):
    def __init__(self, topic: str, instruction=DEFAULT_INSTRUCTION, debug=False):
        super().__init__(instruction, debug)
        berry_message_completer_config = BerrySelectiveMessageCompleter.Config(
            token_completer_config=BusTokenCompleter.Config(
                # assumes you spinned up this engine
                topic_or_snapshot="az://oaidsm2/oaistrawberry2/twapi/mini/e/bmckinzie-nv4-25T-mident-ipb512-spi128-tbv2-run1/policy/step_000320/",
                topic_mode_or_user=topic,
            ),
            completion_params={"temperature": 1, "top_p": 0.995, "max_tokens": 64000},
            renderer="harmony_v4.0.15_berry_v3_1mil_orion_lpe",
        )
        self.completer = berry_message_completer_config.build()

    async def rate_text(self, text: str, rubrics: str):
        convo = chat.Conversation(
            messages=[
                chat.Message.system("You are ChatGPT."),
                chat.Message(
                    role=chat.Role.USER,
                    content=chat.MultimodalText(parts=[self.instruction, rubrics, text]),
                ),
            ]
        )
        completion = self.completer.completion(
            [convo],
            channel_selector=BerryChannel.FINAL_ANSWER,  # pyright: ignore
            valid_channels=[BerryChannel.CHAIN_OF_THOUGHT, BerryChannel.FINAL_ANSWER],  # pyright: ignore
            reward_multiplier=256,  # pyright: ignore
            return_all_messages=True,  # pyright: ignore
        )
        return completion.choices[0].output_conversation.messages[-1].content.parts[0]


def _get_lm(
    model: typing.Literal["neutrino", "gpt-4o-mini", "gpt4o-sonic-training"],
    topic: str | None = None,
) -> lm_utils.LanguageModel:
    if model == "neutrino":
        assert topic is not None
        lm = lm_utils.neutrino_from_completer(
            topic_mode_or_user=topic,
        )
    else:
        lm = lm_utils.lm_from_api(model=model)

    return lm_utils.RetriableLM(
        lm=lm,
        max_retry=3,
        max_concurrency=10,
    )


class GeneralGrader(Grader):
    def __init__(
        self,
        model: typing.Literal["neutrino", "gpt-4o-mini", "gpt4o-sonic-training"],
        instruction=DEFAULT_INSTRUCTION,
        topic: str | None = None,
        debug=False,
    ):
        super().__init__(instruction, debug)
        self.grader = _get_lm(model=model, topic=topic)

    async def rate_text(self, text: str, rubrics: str):
        conv = [
            lm_utils.system_message("You are ChatGPT, a large language model developed by OpenAI."),
            lm_utils.user_message("\n".join([self.instruction, rubrics, text])),
        ]

        ans = await self.grader(conv)
        if not ans or not ans.content:
            raise ValueError("Cannot parse empty message content")
        return ans.content


def get_grader(model: TModelType, topic: str | None = None, debug: bool = False) -> Grader:
    if model == "tb2_with_browse":
        return ToolberryWithBrowseGrader(debug=debug)
    if model == "cb":
        return ChatberryGrader(debug=debug)
    elif model == "nv4":
        return NeutrinoV4Grader(topic=none_throws(topic), debug=debug)
    else:
        return GeneralGrader(model=model, topic=topic, debug=debug)


def build_grader_input(
    record: typing.Any,
    rating_mode: TRatingMode,
) -> typing.Generator[tuple[Query, DocumentWithContext], typing.Any, typing.Any]:
    query = record["query"]
    location = [s.strip() for s in query.get("location", ",,").split(",")]
    q = Query(
        query=query["query"],
        create_timestamp=query.get("timestamp"),
        user_metadata=UserMetadata(
            user_country=location[2],
            user_region=location[1],
            ip_city=location[0],
        ),
    )

    search_result_groups = record["search_result_groups"]
    for group in search_result_groups:
        for page in group["pages"]:
            if rating_mode == "page":
                d = DocumentWithContext(
                    document=Document(
                        url=page.get("url"),
                        title=page.get("title"),
                        pub_date=page.get("pub_date"),
                        content=page.get("content"),
                    ),
                )
                yield q, d
            else:
                for snippet in page.get("snippets", []):
                    d = DocumentWithContext(
                        document=Document(
                            url=page.get("url"),
                            title=page.get("title"),
                            pub_date=page.get("pub_date"),
                            content=snippet,
                        ),
                        context=(
                            page.get("content", "") if rating_mode == "snippet_w_context" else ""
                        ),
                    )
                    yield q, d


def _progress(*, console: rich.console.Console) -> rich.progress.Progress:
    return rich.progress.Progress(
        *rich.progress.Progress.get_default_columns(),
        rich.progress.MofNCompleteColumn(),
        console=console,
    )


async def run(
    *,
    input_path: str,
    output_path: str,
    model: TModelType,
    rating_mode: TRatingMode,
    limit: int | None = None,
    grading_concurrency: int = 20,
    random_seed: int | None = None,
    read_buffer: int | None = None,
    topic: str | None = None,
    debug: bool = False,
) -> None:
    grader = get_grader(model, topic=topic, debug=debug)
    sem = asyncio.Semaphore(grading_concurrency)

    # default buffer to 10x limit
    if limit and not read_buffer:
        read_buffer = limit * 10

    # if no limit, do not need shuffle
    if not limit:
        random_seed = None

    if random_seed is not None:
        random.seed(random_seed)
        # lower bound buffer for shuffling is 2x limit
        assert none_throws(read_buffer) > none_throws(limit) * 2

    with _progress(console=rich.get_console()) as pbar:
        task = pbar.add_task("Grading ...", total=limit)

        async def grade_one(
            q: Query,
            d: DocumentWithContext,
            rubrics: str,
            s: asyncio.Semaphore,
        ):
            async with s:
                res = await grader(q, d, rubrics=rubrics)
                pbar.update(task, advance=1)
                return res

        qds = []
        with oai_json.jsonl_load_stream(input_path) as read_f:
            for record in read_f:
                for query, document in build_grader_input(record, rating_mode):
                    qds.append((query, document))
                    if read_buffer and len(qds) >= read_buffer:
                        break

        if random_seed:
            random.shuffle(qds)

        qds = qds[:limit]
        jobs = []
        for query, document in qds:
            jobs.append(grade_one(query, document, DEFAULT_RUBRICS, sem))
        pbar.update(task, total=len(jobs))

        results = await asyncio.gather(*jobs)

        with oai_json.jsonl_dump_stream(output_path) as write_f:
            for (query, document), result in zip(qds, results, strict=True):
                write_f.write(
                    {
                        "query": query.model_dump_json(),
                        "document": document.model_dump_json(),
                        "grading": result,
                    }
                )

        azv = (
            f", use http://go/azv/{output_path} to view " if output_path.startswith("az://") else ""
        )
        print(f"Dumped data to {output_path}{azv}")


def main(
    input_path: str = "/tmp/dump_feather_results/sample_content_enriched_search_result_groups_dump.jsonl",
    output_path: str = "/tmp/dump_feather_results/graded_search_result_groups_dump_Rel_SamplewC_Accuracy_nv4_wC.jsonl",
    model: TModelType = "nv4",
    rating_mode: TRatingMode = "snippet_w_context",
    limit: int | None = None,
    grading_concurrency: int = 40,
    random_seed: int | None = None,
    read_buffer: int | None = None,
    topic: str = "bwen-citation-neutrinov4-test",
    debug: bool = False,
) -> None:
    asyncio.run(
        run(
            input_path=input_path,
            output_path=output_path,
            model=model,
            rating_mode=rating_mode,
            limit=limit,
            grading_concurrency=grading_concurrency,
            random_seed=random_seed,
            read_buffer=read_buffer,
            topic=topic,
            debug=debug,
        )
    )


if __name__ == "__main__":
    chz.entrypoint(main, allow_hyphens=True)

