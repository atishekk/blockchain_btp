from vm.vm import VM
from vm.request import RequestQueue


class CLI:
    USAGE = """
        setup 
        query
    """

    @classmethod
    def run(cls, ledger: str):
        interface = cls(ledger)
        while True:
            tx = input(">>> ").strip()
            tokens = tx.split(" ")

            match tokens[0].lower():
                case "setup":
                    interface.setup()
                case "query":
                    interface.query()
                case "quit":
                    interface.cleanup()
                    exit(0)
                case _:
                    print(f"Invalid request: '{tx}'")
                    print(cls.USAGE)

    def __init__(self, ledger_file: str):
        self.vm = VM(RequestQueue(ledger_file))

    def cleanup(self):
        print("Clean up")

    def setup(self):
        print("Setup")

    def query(self):
        print("Query")
