from collections import defaultdict
from transactionDataset import TransactionDataset
import itertools

class AprioriAlgorithm:
    def __init__(self, transaction_dataset, min_support):
        self.transaction_dataset = transaction_dataset
        self.min_support = min_support
        self.frequent_itemsets = []

    def run(self):
        self._generate_frequent_itemsets()
        association_rules = self._generate_association_rules(min_confidence=0.5)
        self.print_association_rules(association_rules)

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
                    item_counts[tuple(itemset)] += 1

        for itemset, count in item_counts.items():
            if count >= self.min_support:
                frequent_itemsets.append(list(itemset))

        return frequent_itemsets

    def _generate_association_rules(self, min_confidence):
        association_rules = []
        for itemsets in self.frequent_itemsets[1:]:
            for itemset in itemsets:
                subsets = self._get_subsets(itemset)
                for subset in subsets:
                    antecedent = subset
                    consequent = list(set(itemset) - set(antecedent))
                    confidence = self._calculate_confidence(antecedent, consequent)
                    if confidence >= min_confidence:
                        rule = (antecedent, consequent, confidence)
                        association_rules.append(rule)

        return association_rules

    def _get_subsets(self, itemset):
        subsets = []
        for i in range(1, len(itemset)):
            subsets.extend(itertools.combinations(itemset, i))
        return subsets

    def _calculate_confidence(self, antecedent, consequent):
        combined_itemset_count = self._count_itemset(tuple(antecedent) + tuple(consequent))
        antecedent_count = self._count_itemset(tuple(antecedent))
        return combined_itemset_count / antecedent_count




    def _count_itemset(self, itemset):
        count = 0
        for transaction in self.transaction_dataset.transactions:
            if set(itemset).issubset(transaction):
                count += 1
        return count

    def print_association_rules(self, association_rules):
        print("Association Rules:")
        for rule in association_rules:
            antecedent, consequent, confidence = rule
            print("Antecedent:", antecedent)
            print("Consequent:", consequent)
            print("Confidence:", confidence)
            print()

if __name__ == "__main__":
    # Create a TransactionDataset object
    transaction_dataset = TransactionDataset()

    # Add transactions
    transaction_dataset.add_transaction(["A", "B", "C", "D"])
    transaction_dataset.add_transaction(["B", "C", "D", "E"])
    transaction_dataset.add_transaction(["A", "C", "E"])
    transaction_dataset.add_transaction(["B", "D"])
    transaction_dataset.add_transaction(["A", "C", "D"])

    # Create an AprioriAlgorithm object
    apriori = AprioriAlgorithm(transaction_dataset, min_support=3)

    # Run the Apriori algorithm
    apriori.run()

    # Generate association rules with minimum confidence
    min_confidence = 0.5
    association_rules = apriori._generate_association_rules(min_confidence)

    # Print the frequent itemsets
    frequent_itemsets = apriori.frequent_itemsets
    print("Frequent Itemsets:")
    for k, itemsets in enumerate(frequent_itemsets):
        print("k =", k + 1, ":", itemsets)

    # Print the association rules
    print("Association Rules:")
    for rule in association_rules:
        print(rule)
