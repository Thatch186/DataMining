from collections import defaultdict
from transactionDataset import TransactionDataset

class FPTree:
    class Node:
        def __init__(self, item, count=0):
            self.item = item
            self.count = count
            self.children = []
            self.parent = None

    def __init__(self, transaction_dataset):
        self.root = self.Node(None)
        self.header_table = defaultdict(list)
        self.build_tree(transaction_dataset)

    def build_tree(self, transaction_dataset):
        item_counts = defaultdict(int)
        for transaction in transaction_dataset.transactions:
            for item in transaction:
                item_counts[item] += 1

        for transaction in transaction_dataset.transactions:
            sorted_items = sorted(transaction, key=lambda x: item_counts[x], reverse=True)
            self.insert_transaction(sorted_items)

    def insert_transaction(self, sorted_items):
        current_node = self.root
        for item in sorted_items:
            child_node = self._find_child_node(current_node, item)
            if child_node is None:
                child_node = self.Node(item)
                child_node.parent = current_node
                current_node.children.append(child_node)
                self.header_table[item].append(child_node)
            child_node.count += 1
            current_node = child_node

    def _find_child_node(self, parent_node, item):
        for child_node in parent_node.children:
            if child_node.item == item:
                return child_node
        return None

    def print_tree_root(self):
        print("FP Tree Root:", self.root.item)

    def print_header_table(self):
        print("Header Table:")
        for item, nodes in self.header_table.items():
            node_items = [node.item for node in nodes]
            print(f"Item: {item}, Nodes: {node_items}")


if __name__ == "__main__":
    # Create a TransactionDataset object
    transaction_dataset = TransactionDataset()

    # Add transactions
    transaction_dataset.add_transaction(["item1", "item2", "item3"])
    transaction_dataset.add_transaction(["item2", "item3", "item4"])
    transaction_dataset.add_transaction(["item1", "item3", "item4"])

    # Create an FPTree object
    fp_tree = FPTree(transaction_dataset)

    # Print the FP Tree root
    fp_tree.print_tree_root()

    # Print the header table
    fp_tree.print_header_table()
