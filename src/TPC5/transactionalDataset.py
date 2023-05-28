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




