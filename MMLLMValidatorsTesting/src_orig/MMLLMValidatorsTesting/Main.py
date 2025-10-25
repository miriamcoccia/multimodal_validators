import argparse
import os
import uuid
from config import settings
from .LLMValidatorsTesting import LLMValidatorsTesting
from .Prompt import PromptID

print("LLMValidatorsTesting - Main.py")


def list_of_strings(arg):
    return arg.split(",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="LLMValidatorsTesting",
        description="Testing framework to assess the performance of LLM-based Validators.",
        epilog="",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_filename",
        default=None,
        help="Input .csv file. Defaults to the path in config.toml.",
        metavar="INPUT",
    )

    parser.add_argument(
        "-d",
        "--dir",
        "-o",
        "--out",
        dest="results_dir",
        default=None,
        help="Output directory. Defaults to 'results/<uuid>' using path from config.toml.",
        metavar="DIR",
    )

    parser.add_argument(
        "-N",
        "-n",
        "--max",
        dest="MAX",
        type=int,
        default=0,
        help="Maximum number of questions analysed from the dataset.",
        metavar="MAX",
    )

    parser.add_argument(
        "-r",
        "--random",
        dest="random_choice",
        action="store_true",
        help="Random order of analysed questions.",
    )

    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Run in multimodal mode for models that accept images.",
    )

    parser.add_argument(
        "-m",
        "--models",
        type=list_of_strings,
        dest="models_list",
        default="",
        help="List the LLMs to run.",
        metavar="MODELS",
    )

    parser.add_argument(
        "-sw",
        "--starts-with",
        type=str,
        dest="models_prefix",
        default=None,
        help="Selects all LLMs starting with the <prefix>.",
        metavar="PREFIX",
    )

    parser.add_argument(
        "-p",
        "--prompts",
        type=list_of_strings,
        dest="prompts_list",
        default="",
        help="List the prompts to use.",
        metavar="PROMPTS",
    )

    parser.add_argument(
        "-ll",
        "--llms",
        "--llm-list",
        dest="list_llms",
        action="store_true",
        help="List the supported LLMs.",
    )

    parser.add_argument(
        "-pl",
        "--prompt-list",
        dest="list_prompts",
        action="store_true",
        help="List the available prompts.",
    )

    args = parser.parse_args()

    input_file = args.input_filename or settings.get("paths", {}).get("input_data_csv")

    results_path = args.results_dir
    if results_path is None:
        base_results_dir = settings.get("paths", {}).get("results_dir", "results/")
        results_path = os.path.join(base_results_dir, str(uuid.uuid4()))

    print(args)
    if not input_file:
        parser.error(
            "Input filename is mandatory (provide via --input or in config.toml)."
        )
    elif not os.path.exists(input_file):
        parser.error(f"Input file '{input_file}' does not exist.")
    else:
        # Create results directory if it doesn't exist
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        LLMrunner = LLMValidatorsTesting(
            input_filename=input_file,
            results_dir=results_path,
            MAX=args.MAX,
            random_choice=args.random_choice,
            multimodal=args.multimodal,
        )

        if args.list_llms:
            LLMrunner.llm_service.print_supported_llms()
        if args.list_prompts:
            PromptID.print_supported_prompts()

        models = []
        if args.models_prefix is not None:
            models = LLMrunner.llm_service.get_model_ids_startswith(args.models_prefix)
            if not models:
                parser.error("Invalid models prefix.")
        elif args.models_list and args.models_list != [""]:
            all_supported_models = LLMrunner.llm_service.get_all_models()
            models = [m for m in args.models_list if m in all_supported_models]
        else:
            models = LLMrunner.llm_service.get_all_models()

        if not models:
            parser.error("No valid models were selected to run.")

        prompt_IDs = []
        if args.prompts_list and args.prompts_list != [""]:
            all_prompt_names = [p.name for p in PromptID.all()]
            for p_name in args.prompts_list:
                if p_name in all_prompt_names:
                    prompt_IDs.append(PromptID[p_name])
        else:
            prompt_IDs = PromptID.all()

        if not prompt_IDs:
            parser.error("No valid prompts were selected to run.")

        LLMrunner.run_per_qid(prompt_ids=prompt_IDs, model_ids=models)
