"""RAGAS evaluation script for RAG pipeline quality assessment.

Measures:
  - context_precision: Are retrieved documents relevant to the question?
  - context_recall:    Is the ground truth covered by retrieved contexts?
  - faithfulness:      Is the generated answer grounded in the contexts?

Usage:
  python -m gustobot.scripts.evaluate_rag
  python -m gustobot.scripts.evaluate_rag --mode hybrid
  python -m gustobot.scripts.evaluate_rag --mode hybrid --data data/evaluation/test_questions.json
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from loguru import logger


def _try_import_ragas():
    """Try to import ragas, return None if unavailable."""
    try:
        import ragas  # noqa: F401
        from datasets import Dataset  # noqa: F401
        return True
    except ImportError:
        return False


def load_test_data(path: str) -> List[Dict[str, str]]:
    """Load test questions from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d test questions from %s", len(data), path)
    return data


def parse_args() -> Dict[str, Any]:
    """Simple arg parsing without argparse dependency."""
    args = {
        "mode": "dense",
        "data": "data/evaluation/test_questions.json",
        "output": None,
        "limit": None,
    }
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--mode" and i + 1 < len(argv):
            args["mode"] = argv[i + 1]
        elif arg == "--data" and i + 1 < len(argv):
            args["data"] = argv[i + 1]
        elif arg == "--output" and i + 1 < len(argv):
            args["output"] = argv[i + 1]
        elif arg == "--limit" and i + 1 < len(argv):
            args["limit"] = int(argv[i + 1])
    return args


async def run_retrieval(
    question: str,
    service,
    top_k: int = 5,
) -> List[str]:
    """Run retrieval via KnowledgeService and return context texts."""
    results = await service.search(query=question, top_k=top_k)
    return [r.get("content", "") for r in results]


async def generate_answer(
    question: str,
    contexts: List[str],
) -> str:
    """Generate an answer from retrieved contexts using the project's LLM."""
    from langchain_openai import ChatOpenAI
    from gustobot.config import settings

    model = ChatOpenAI(
        openai_api_key=settings.LLM_API_KEY,
        model_name=settings.LLM_MODEL,
        openai_api_base=settings.LLM_BASE_URL,
        temperature=0.3,
    )

    context_text = "\n\n".join(
        f"[{i+1}] {ctx[:2000]}" for i, ctx in enumerate(contexts)
    )
    prompt = f"""请根据以下检索到的知识库内容回答问题。

检索到的知识库内容：
{context_text}

问题：{question}

请基于以上知识库内容给出回答。如果知识库中没有相关信息，请说明。"""

    response = await model.ainvoke([{"role": "user", "content": prompt}])
    return response.content


def print_report(
    results: Dict[str, Any],
    mode_name: str,
) -> None:
    """Print evaluation results."""
    print(f"\n{'='*60}")
    print(f"  RAG Evaluation Report — Mode: {mode_name}")
    print(f"{'='*60}")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"  {metric:30s}: {value:.4f}")
        else:
            print(f"  {metric:30s}: {value}")
    print(f"{'='*60}\n")


async def main():
    args = parse_args()
    mode = args["mode"]
    data_path = args["data"]
    output_path = args["output"]
    question_limit = args["limit"]

    if not _try_import_ragas():
        print("Error: ragas and datasets are required. Install with:")
        print("  pip install ragas datasets")
        sys.exit(1)

    import ragas
    from datasets import Dataset

    try:
        from ragas.metrics.collections import faithfulness, context_precision
    except ImportError:
        from ragas.metrics import faithfulness, context_precision

    # Optionally include context_recall (needs ground_truth)
    include_recall = False
    try:
        from ragas.metrics.collections import context_recall
        include_recall = True
    except ImportError:
        try:
            from ragas.metrics import context_recall
            include_recall = True
        except ImportError:
            logger.warning("context_recall metric not available in this ragas version")

    # Load test data
    test_data = load_test_data(data_path)
    if question_limit:
        test_data = test_data[:question_limit]

    # Build KnowledgeService
    if mode == "hybrid":
        os.environ["HYBRID_SEARCH_ENABLED"] = "true"
        logger.info("Running in HYBRID mode (dense ANN + sparse BM25 + RRF)")
    else:
        os.environ["HYBRID_SEARCH_ENABLED"] = "false"
        logger.info("Running in DENSE-ONLY mode (Milvus ANN)")

    from gustobot.infrastructure.knowledge import KnowledgeService

    service = KnowledgeService()

    # Run retrieval and answer generation
    questions = []
    contexts_list = []
    answers = []
    ground_truths = []

    for i, item in enumerate(test_data):
        question = item["question"]
        gt = item.get("ground_truth", "")
        logger.info("[%d/%d] Processing: %s", i + 1, len(test_data), question[:50])

        contexts = await run_retrieval(question, service)
        answer = await generate_answer(question, contexts)

        questions.append(question)
        contexts_list.append(contexts)
        answers.append(answer)
        ground_truths.append(gt)

    # Build RAGAS dataset
    eval_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
    }
    if any(ground_truths):
        eval_data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(eval_data)

    # Compute metrics
    ragas_version = tuple(int(x) for x in ragas.__version__.split(".")[:2])

    metrics = [faithfulness, context_precision]
    if include_recall and any(ground_truths):
        metrics.append(context_recall)

    if ragas_version >= (0, 4):
        # ragas 0.4+ API
        result = ragas.evaluate(dataset, metrics=metrics)
        scores = {m: result[m] for m in [m.name for m in metrics]}
    else:
        # ragas 0.1-0.3 API
        result = ragas.evaluate(dataset, metrics=metrics)
        scores = {m.__name__ if hasattr(m, '__name__') else str(m): result[m] for m in metrics}

    print_report(scores, mode.upper())

    # Save results
    if output_path:
        output = {
            "mode": mode,
            "metrics": scores,
            "details": [
                {
                    "question": q,
                    "context_count": len(c),
                    "answer_length": len(a),
                }
                for q, c, a in zip(questions, contexts_list, answers)
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        logger.info("Results saved to %s", output_path)

    return scores


if __name__ == "__main__":
    asyncio.run(main())
