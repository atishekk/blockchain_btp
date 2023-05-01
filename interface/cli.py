from vm.vm import VM
from vm.request import RequestQueue
from typing import List
from pathlib import Path

from cryptography.fernet import Fernet


class CLI:
    USAGE = """
        setup <model-name>
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
                    interface.setup(tokens)

                case "query":
                    interface.query(tokens)

                case "utils":
                    if len(tokens) < 4:
                        print(f"Invalid request: '{tx}'")
                        print(cls.USAGE)
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
                    print(f"Invalid request: '{tx}'")
                    print(cls.USAGE)

    def __init__(self, ledger_file: Path):
        self.vm = VM(RequestQueue(ledger_file))

    def utils_build(self, tokens: List[str]):
        model_name = tokens[2]
        if model_name not in self.vm.MODELS:
            print(f"Invalid model name: {model_name}")
            return

        model_cls = self.vm.MODELS[model_name]
        model_state_file = Path(tokens[3])
        block_model = model_cls.new(model_state_file)

        out_file_name = Path(model_state_file.stem + ".sqlite")
        model_cls.encode(block_model, out_file_name)

        fernet_key = Fernet.generate_key()
        encryptor = Fernet(fernet_key)
        with open(out_file_name, "rb") as dict_model:
            data = dict_model.read()

        en_data = encryptor.encrypt(data)

        with open(out_file_name, "wb") as model_file:
            model_file.write(en_data)

        print(f"MODEL file written at {out_file_name} ")
        print(f"Encryption key: {fernet_key.hex()}")

        return

    def utils_publish(self):
        pass

    def cleanup(self):
        print("Clean up")

    def setup(self, tokens: List[str]):
        print("Setup")

    def query(self, tokens: List[str]):
        print("Query")
