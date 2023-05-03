from vm.vm import VM
from vm.request import RequestQueue, SetupInput
from typing import List
from pathlib import Path

from cryptography.fernet import Fernet


class CLI:
    USAGE = """
        setup <model-name> <model-file> <fernet-key>
        query <image-path>
        utils build <model-name> <model-state-file>
        utils publish <model-file> <encryption-key>
    """

    @classmethod
    def run(cls, ledger: Path):
        interface = cls(ledger)
        while True:
            tx = input(">>> ").strip()
            tokens = tx.split(" ")

            match tokens[0].lower():
                case "setup":
                    if len(tokens) < 4:
                        interface.print_usage(tx)
                        continue
                    interface.setup(tokens)

                case "query":
                    if len(tokens) < 2:
                        interface.print_usage(tx)
                        continue
                    interface.query(tokens)

                case "utils":
                    if len(tokens) < 4:
                        interface.print_usage(tx)
                        continue

                    match tokens[1].lower():
                        case "build":
                            interface.utils_build(tokens)
                        case "publish":
                            interface.utils_publish()

                case "quit":
                    interface.cleanup()
                    exit(0)

                case _:
                    interface.print_usage(tx)

    def __init__(self, ledger_file: Path):
        self.vm = VM(RequestQueue(ledger_file))

    def setup(self, tokens: List[str]):
        model_name = tokens[1]
        model_file = Path(tokens[2])
        fernet_key = tokens[3]

        s_input = SetupInput(model_name, model_file, fernet_key)
        s_input.validate(self.vm)
        self.vm.setup(s_input)

    def query(self, tokens: List[str]):
        print("Query")

    def utils_build(self, tokens: List[str]):
        model_name = tokens[2]
        if model_name not in self.vm.MODELS:
            print(f"Invalid model name: {model_name}")
            return

        model_cls = self.vm.MODELS[model_name]
        model_state_file = Path(tokens[3])
        block_model = model_cls.new(model_state_file)

        out_file_name = Path(model_state_file.stem + ".sqlite")
        fernet_key = model_cls.encode(block_model, out_file_name)

        print(f"MODEL file written at {out_file_name} ")
        print(f"Encryption key: {fernet_key}")
        return

    def utils_publish(self):
        pass

    def cleanup(self):
        print("Clean up")

    def print_usage(self, req: str):
        print(f"Invalid request: '{req}'")
        print(self.USAGE)
