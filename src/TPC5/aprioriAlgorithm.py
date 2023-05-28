class AprioriAlgorithm:
    def __init__(self, transaction_dataset, min_support):
        self.transaction_dataset = transaction_dataset
        self.min_support = min_support
        self.frequent_itemsets = []

    def run(self):
        self._generate_frequent_itemsets()
        self._generate_association_rules()

    def _generate_frequent_itemsets(self):
        frequent_items = self.transaction_dataset.get_frequent_items(self.min_support)
        self.frequent_itemsets.append(frequent_items)

        k = 2
        while True:
            candidate_itemsets = self._generate_candidate_itemsets(self.frequent_itemsets[k - 2], k)
            frequent_itemsets = self._filter_frequent_itemsets(candidate_itemsets)
            if not frequent_itemsets:
                break
            self.frequent_itemsets.append(frequent_itemsets)
            k += 1

    def _generate_candidate_itemsets(self, frequent_itemsets, k):
        candidate_itemsets = []
        for i in range(len(frequent_itemsets)):
            for j in range(i + 1, len(frequent_itemsets)):
                itemset1 = frequent_itemsets[i]
                itemset2 = frequent_itemsets[j]
                if itemset1[:k - 2] == itemset2[:k - 2]:
                    candidate_itemsets.append(sorted(set(itemset1).union(itemset2)))
        return candidate_itemsets

    def _filter_frequent_itemsets(self, candidate_itemsets):
        frequent_itemsets = []
        item_counts = defaultdict(int)
        for transaction in self.transaction_dataset.transactions:
            for itemset in candidate_itemsets:
                if set(itemset).issubset(transaction):
                    item_counts[tuple(itemset)]
