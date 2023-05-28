from collections import defaultdict

class TransactionDataset:
    def __init__(self):
        self.transactions = []

    def add_transaction(self, transaction):
        self.transactions.append(transaction)

    def get_frequent_items(self, min_support):
        item_counts = defaultdict(int)
        for transaction in self.transactions:
            for item in transaction:
                item_counts[item] += 1

        frequent_items = [item for item, count in item_counts.items() if count >= min_support]
        frequent_items.sort(key=lambda x: item_counts[x], reverse=True)
        return frequent_items


if __name__ == "__main__":
    # Create a TransactionDataset object
    dataset = TransactionDataset()

    # Add transactions
    dataset.add_transaction(["item1", "item2", "item3"])
    dataset.add_transaction(["item2", "item3", "item4"])
    dataset.add_transaction(["item1", "item3", "item4"])

    # Get frequent items with minimum support of 2
    frequent_items = dataset.get_frequent_items(min_support=2)
    print("Frequent Items:", frequent_items)

