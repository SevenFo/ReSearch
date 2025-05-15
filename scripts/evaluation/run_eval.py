from flashrag.config import Config
from flashrag.utils import get_dataset
import argparse


def naive(args, config_dict):
    from flashrag.pipeline import SequentialPipeline

    # preparation
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    pipeline = SequentialPipeline(config)

    result = pipeline.run(test_data)


def zero_shot(args, config_dict):
    from flashrag.pipeline import SequentialPipeline

    # preparation
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import SequentialPipeline
    from flashrag.prompt import PromptTemplate

    templete = PromptTemplate(
        config=config,
        system_prompt="Answer the question based on your own knowledge. Only give me the answer and do not output any other words.",
        user_prompt="Question: {question}",
    )
    pipeline = SequentialPipeline(config, templete)
    result = pipeline.naive_run(test_data)


def iterretgen(args, config_dict):
    """
    Reference:
        Zhihong Shao et al. "Enhancing Retrieval-Augmented Large Language Models with Iterative
                            Retrieval-Generation Synergy"
        in EMNLP Findings 2023.

        Zhangyin Feng et al. "Retrieval-Generation Synergy Augmented Large Language Models"
        in EMNLP Findings 2023.
    """
    iter_num = 3

    # preparation
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import IterativePipeline

    pipeline = IterativePipeline(config, iter_num=iter_num)
    result = pipeline.run(test_data)


def ircot(args, config_dict):
    """
    Reference:
        Harsh Trivedi et al. "Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions"
        in ACL 2023
    """
    from flashrag.pipeline import IRCOTPipeline

    # preparation
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]
    print(config["generator_model_path"])
    pipeline = IRCOTPipeline(config, max_iter=5)

    result = pipeline.run(test_data)


def research(args, config_dict):
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import ReSearchPipeline

    pipeline = ReSearchPipeline(config, apply_chat=args.apply_chat)
    result = pipeline.run(test_data)


def r1r(args, config_dict):
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import R1SearcherPipeline

    pipeline = R1SearcherPipeline(config, apply_chat=args.apply_chat)
    result = pipeline.run(test_data)


def research_crag(args, config_dict):
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import ReSearchPipeline
    from flashrag.retrieve_evaluator.crag_evaluator import RemoteCRAGEvaluator
    from flashrag.retrieve_evaluator.ollama_evaluator import OllamaDirectRAGEvaluator

    if config["retriever_evaluate_model_type"] == "qwen":
        remote_crag = OllamaDirectRAGEvaluator(
            "qwen2.5:7b",
            config["remote_crag_evaluator_url"],
        )
    elif config["retriever_evaluate_model_type"] == "t5":
        remote_crag = RemoteCRAGEvaluator(
            {"remote_evaluator_url": config["remote_crag_evaluator_url"]}
        )
    else:
        print(f"Unknown model type: {config['retriever_evaluate_model_type']}")
        return
    pipeline = ReSearchPipeline(
        config, apply_chat=args.apply_chat, evaluator=remote_crag
    )
    result = pipeline.run(test_data)


def r1r_crag(args, config_dict):
    config = Config(args.config_path, config_dict)
    all_split = get_dataset(config)
    test_data = all_split[args.split]

    from flashrag.pipeline import R1SearcherPipeline
    from flashrag.retrieve_evaluator.crag_evaluator import RemoteCRAGEvaluator
    from flashrag.retrieve_evaluator.ollama_evaluator import OllamaDirectRAGEvaluator

    if config["retriever_evaluate_model_type"] == "qwen":
        remote_crag = OllamaDirectRAGEvaluator(
            "qwen2.5:7b",
            config["remote_crag_evaluator_url"],
        )
    elif config["retriever_evaluate_model_type"] == "t5":
        remote_crag = RemoteCRAGEvaluator(
            {"remote_evaluator_url": config["remote_crag_evaluator_url"]}
        )
    else:
        print(f"Unknown model type: {config['retriever_evaluate_model_type']}")
        return
    pipeline = R1SearcherPipeline(
        config, apply_chat=args.apply_chat, evaluator=remote_crag
    )
    result = pipeline.run(test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running exp")
    parser.add_argument("--config_path", type=str, default="./eval_config.yaml")
    parser.add_argument("--method_name", type=str, default="research")
    parser.add_argument("--data_dir", type=str, default="your-data-dir")
    parser.add_argument("--dataset_name", type=str, default="bamboogle")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--save_dir", type=str, default="your-save-dir")
    parser.add_argument(
        "--save_note", type=str, default="your-save-note-for-identification"
    )
    parser.add_argument("--sgl_remote_url", type=str, default="your-sgl-remote-url")
    parser.add_argument(
        "--remote_retriever_url", type=str, default="your-remote-retriever-url"
    )
    parser.add_argument("--generator_model", type=str, default="your-local-model-path")
    parser.add_argument("--apply_chat", action="store_true")
    parser.add_argument(
        "--retrieve_evaluate_strategy",
        type=str,
        default="separate_filter",
        help="The strategy to evaluate the retrieval results: 'separate_filter' or 'full_filter' or 'separate_replace' or 'full_replace'.",
    )
    parser.add_argument(
        "--evaluate_on_subject",
        action="store_true",
        help="Whether to evaluate on subject.",
    )
    parser.add_argument(
        "--evaluate_model_type",
        type=str,
        default="qwen",
        help="The model type for evaluation. Options: 'qwen', 't5'",
    )
    parser.add_argument(
        "--remote_crag_evaluator_url",
        type=str,
        default="http://localhost:10098",
        help="URL of the remote retriever service.",
    )
    parser.add_argument(
        "--upper_threshold",
        type=float,
        default=0.5,
        help="Upper threshold for CRAG evaluator.",
    )
    print("Arguments:")
    print(parser.parse_args())
    func_dict = {
        "naive": naive,
        "zero-shot": zero_shot,
        "iterretgen": iterretgen,
        "ircot": ircot,
        "research": research,
        "r1r": r1r,
        "r1r_crag": r1r_crag,
        "research_crag": research_crag,
    }

    args = parser.parse_args()

    config_dict = {
        "data_dir": args.data_dir,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "save_dir": args.save_dir,
        "save_note": args.save_note if args.save_note else args.method_name,
        "sgl_remote_url": args.sgl_remote_url,
        "remote_retriever_url": args.remote_retriever_url,
        "generator_model": args.generator_model,
        "retrieve_evaluate_strategy": args.retrieve_evaluate_strategy,
        "evaluate_on_subject": args.evaluate_on_subject,
        "remote_crag_evaluator_url": args.remote_crag_evaluator_url,
        "upper_threshold": args.upper_threshold,
        "retriever_evaluate_model_type": args.evaluate_model_type,
    }

    print(config_dict)

    func = func_dict[args.method_name]
    func(args, config_dict)
