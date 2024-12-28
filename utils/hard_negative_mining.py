from pytorch_metric_learning.miners import BaseMiner
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

class ExamplePairMiner(BaseMiner):
    def __init__(self, margin=0.1, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = self.distance(embeddings, ref_emb)
        a1, p, a2, n = lmu.get_all_pairs_indices(labels, ref_labels)
        pos_pairs = mat[a1, p]
        neg_pairs = mat[a2, n]
        pos_mask = (
            pos_pairs < self.margin
            if self.distance.is_inverted
            else pos_pairs > self.margin
        )
        neg_mask = (
            neg_pairs > self.margin
            if self.distance.is_inverted
            else neg_pairs < self.margin
        )
        return a1[pos_mask], p[pos_mask], a2[neg_mask], n[neg_mask]