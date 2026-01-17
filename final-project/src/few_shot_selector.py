"""
Dynamic few-shot example selection for LLM reranking.

Best practices implemented:
- Embedding-based similarity to select relevant examples
- Hybrid approach: 1 static anchor + 2-3 dynamic examples
- Diversity: avoid near-duplicate examples
- Pool of real ROBUST04 examples covering different difficulty levels
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class FewShotExample:
    """A single few-shot example with query, documents, and ranking."""
    qid: str
    query: str
    difficulty: str  # "easy", "medium", "hard"
    documents: list[dict]  # [{"text": ..., "relevant": True/False}, ...]
    ranking: list[int]  # Correct ranking (1-indexed)
    reasoning: str  # Brief reasoning
    embedding: np.ndarray | None = None


# Real ROBUST04 examples from the training set
# Each covers a different difficulty level and topic type
FEW_SHOT_POOL = [
    # EASY queries (many relevant docs) - broad topics
    FewShotExample(
        qid="301",
        query="international organized crime",
        difficulty="easy",
        documents=[
            {"text": "Colombian Prosecutor General Gustavo de Greiff said the U.S. Government is not interested in supplying evidence to condemn the chiefs of the drug trafficking mafia because it does not trust Colombian justice. De Greiff reaffirmed difficulties collecting evidence against the main 'capos' of the Cali cartel.", "relevant": True},
            {"text": "State of the Nation Address by President Guillermo Endara at the Legislative Palace in Panama City: Your Excellencies, president of the Legislative Assembly; president of the Supreme Court of Justice; Vice President Guillermo Ford Boyd; Mr. Ministers and Deputy Ministers...", "relevant": False},
            {"text": "The International Drugs Control Board has charged drug traffickers are continuously infiltrating the Colombian Government and Congress. The board has information about visits attorneys of the Cali Cartel have made to Congress during discussions of the Penal Procedures Code.", "relevant": True},
        ],
        ranking=[1, 3, 2],
        reasoning="Doc 1: HIGHLY - drug trafficking mafia, Cali cartel, international cooperation. Doc 3: HIGHLY - Cali Cartel infiltrating government, international board. Doc 2: NOT - government address, no crime content.",
    ),
    FewShotExample(
        qid="306",
        query="african civilian deaths",
        difficulty="easy",
        documents=[
            {"text": "The United Nations has reported that over 500,000 civilians have died in the Rwandan genocide. Hutu militias systematically massacred Tutsi populations while the international community failed to intervene.", "relevant": True},
            {"text": "Agricultural production in sub-Saharan Africa has increased by 3% this year, according to the World Bank. Coffee exports remain the primary source of foreign currency.", "relevant": False},
            {"text": "Relief agencies estimate 2 million refugees have fled the violence in Burundi, with thousands killed in ethnic clashes between Hutu and Tutsi groups.", "relevant": True},
        ],
        ranking=[1, 3, 2],
        reasoning="Doc 1: HIGHLY - Rwandan genocide, civilian massacres. Doc 3: HIGHLY - Burundi violence, ethnic killings. Doc 2: NOT - agriculture, no deaths mentioned.",
    ),
    
    # HARD queries (few relevant docs) - narrow topics
    FewShotExample(
        qid="309",
        query="rap crime",
        difficulty="hard",
        documents=[
            {"text": "Two young men did not kill themselves because they heard alleged subliminal messages in the heavy metal music of Judas Priest, a judge in Reno ruled. The case examined whether rock music could cause violent behavior.", "relevant": True},
            {"text": "Speech by President Wasmosy before Congress giving account of executive branch performance for the past 200 days of administration. Discussion of economic policy and governmental reforms.", "relevant": False},
            {"text": "Rock musicians like Ozzy Osbourne are no more responsible than Goethe or Shakespeare for encouraging suicides. The artist was obsessed with hell, death and Satan. One of the works that made him famous inspired suicides.", "relevant": True},
        ],
        ranking=[1, 3, 2],
        reasoning="Doc 1: RELEVANT - music causing crime/violence, legal case. Doc 3: RELEVANT - music and suicide/violence connection. Doc 2: NOT - government speech, completely off-topic.",
    ),
    FewShotExample(
        qid="308",
        query="implant dentistry",
        difficulty="hard",
        documents=[
            {"text": "Dental implants have revolutionized tooth replacement. Titanium posts are surgically placed into the jawbone, where osseointegration allows them to fuse with bone tissue over several months.", "relevant": True},
            {"text": "The American Dental Association reported increased demand for cosmetic procedures, including teeth whitening and veneers. Consumer spending on dental care rose 5% last year.", "relevant": False},
            {"text": "New techniques in oral surgery allow periodontists to place endosseous implants with minimal invasiveness. Success rates exceed 95% for properly selected patients.", "relevant": True},
        ],
        ranking=[1, 3, 2],
        reasoning="Doc 1: HIGHLY - dental implants, titanium, osseointegration. Doc 3: HIGHLY - endosseous implants, oral surgery. Doc 2: NOT - cosmetic dentistry, no implants.",
    ),
    
    # MEDIUM queries (moderate relevant docs)
    FewShotExample(
        qid="328",
        query="pope beatifications",
        difficulty="medium",
        documents=[
            {"text": "The arcane process of beatification has become the subject of controversy. Before an expected crowd of 200,000 pilgrims at St Peter's in Rome, Pope John Paul II will beatify the founder of Opus Dei, moving him one step closer to sainthood.", "relevant": True},
            {"text": "The members of the Haitian Episcopal Conference will go to Rome to meet with Pope John Paul II. Relations between the Aristide government and the Vatican have always been strained.", "relevant": False},
            {"text": "Fifty years after Auschwitz, the camp where Nazis killed a million people serves as memorial, museum, and tourist attraction. Some victims have been considered for beatification as martyrs.", "relevant": True},
        ],
        ranking=[1, 3, 2],
        reasoning="Doc 1: HIGHLY - beatification ceremony, sainthood process, Pope John Paul II. Doc 3: MEDIUM - mentions beatification of martyrs but focus is Auschwitz tourism. Doc 2: NOT - Pope meeting but no beatification.",
    ),
    FewShotExample(
        qid="340",
        query="land mine ban",
        difficulty="medium",
        documents=[
            {"text": "The Ottawa Treaty to ban antipersonnel mines was signed by 122 countries. Humanitarian organizations celebrated the agreement as a victory for civilian protection in conflict zones.", "relevant": True},
            {"text": "Military exercises in Eastern Europe included demonstrations of new tank technology. NATO forces practiced coordinated maneuvers across difficult terrain.", "relevant": False},
            {"text": "Cambodia remains one of the most heavily mined countries in the world. International demining efforts have cleared thousands of acres but millions of mines remain buried.", "relevant": True},
        ],
        ranking=[1, 3, 2],
        reasoning="Doc 1: HIGHLY - Ottawa Treaty, mine ban, humanitarian agreement. Doc 3: HIGHLY - demining efforts, mine clearance. Doc 2: NOT - military exercises, no mines mentioned.",
    ),
    FewShotExample(
        qid="343",
        query="police deaths",
        difficulty="medium",
        documents=[
            {"text": "Officer Michael Johnson was shot and killed during a traffic stop in Los Angeles. He was a 15-year veteran of the LAPD and leaves behind a wife and two children.", "relevant": True},
            {"text": "The city council approved a new budget for police equipment, including updated patrol vehicles and communication systems. Officers welcomed the modernization efforts.", "relevant": False},
            {"text": "Three deputies were ambushed while responding to a domestic disturbance call. The FBI is investigating the incident as a targeted attack on law enforcement.", "relevant": True},
        ],
        ranking=[1, 3, 2],
        reasoning="Doc 1: HIGHLY - officer killed, line of duty death. Doc 3: HIGHLY - deputies ambushed, officers killed. Doc 2: NOT - police budget, no deaths.",
    ),
    
    # Additional topic diversity
    FewShotExample(
        qid="400",
        query="amazon rain forest",
        difficulty="easy",
        documents=[
            {"text": "Deforestation in the Brazilian Amazon reached record levels this year. Satellite images show massive fires clearing land for cattle ranching and soybean farming.", "relevant": True},
            {"text": "The technology company Amazon announced new fulfillment centers in South America. E-commerce growth in the region has driven expansion of logistics infrastructure.", "relevant": False},
            {"text": "Indigenous communities in the Amazon basin face increasing threats from illegal logging. Environmental groups are working to protect traditional territories and biodiversity.", "relevant": True},
        ],
        ranking=[1, 3, 2],
        reasoning="Doc 1: HIGHLY - Amazon deforestation, fires, clearing. Doc 3: HIGHLY - Amazon indigenous communities, logging threats. Doc 2: NOT - Amazon company, not rainforest.",
    ),
    FewShotExample(
        qid="365",
        query="el nino",
        difficulty="medium",
        documents=[
            {"text": "El Nino conditions in the Pacific Ocean have caused severe drought in Australia and flooding in Peru. Scientists predict the climate pattern will persist through spring.", "relevant": True},
            {"text": "Weather forecasters issued warnings for thunderstorms across the Midwest. Temperatures are expected to remain above average for the next week.", "relevant": False},
            {"text": "The 1997-98 El Nino was one of the strongest on record, causing $33 billion in damage worldwide. Ocean temperatures rose 5 degrees above normal.", "relevant": True},
        ],
        ranking=[1, 3, 2],
        reasoning="Doc 1: HIGHLY - El Nino conditions, drought, flooding. Doc 3: HIGHLY - 1997-98 El Nino, strong event. Doc 2: NOT - general weather, no El Nino.",
    ),
]


class DynamicFewShotSelector:
    """
    Selects few-shot examples based on query similarity.
    
    Strategy:
    - 1 static anchor example (always included for consistency)
    - 2 dynamic examples selected by embedding similarity
    - Ensures diversity (different difficulties, different topics)
    """
    
    def __init__(
        self,
        pool: list[FewShotExample] = None,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        num_dynamic: int = 2,
        anchor_qid: str = "301",  # Default anchor: international organized crime (easy, broad)
    ):
        self.pool = pool or FEW_SHOT_POOL
        self.embedding_model = embedding_model
        self.num_dynamic = num_dynamic
        self.anchor_qid = anchor_qid
        
        self._encoder = None
        self._embeddings = None
    
    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.embedding_model)
        return self._encoder
    
    def _ensure_embeddings(self):
        """Compute embeddings for all examples in the pool."""
        if self._embeddings is not None:
            return
        
        encoder = self._get_encoder()
        queries = [ex.query for ex in self.pool]
        self._embeddings = encoder.encode(queries, normalize_embeddings=True)
        
        for i, ex in enumerate(self.pool):
            ex.embedding = self._embeddings[i]
    
    def select_examples(
        self,
        query: str,
        exclude_qids: set[str] = None,
    ) -> list[FewShotExample]:
        """
        Select few-shot examples for a given query.
        
        Returns:
            List of FewShotExample objects (anchor + dynamic)
        """
        self._ensure_embeddings()
        exclude_qids = exclude_qids or set()
        
        # Always include anchor
        anchor = next((ex for ex in self.pool if ex.qid == self.anchor_qid), self.pool[0])
        selected = [anchor]
        used_qids = {anchor.qid} | exclude_qids
        
        # Compute query embedding
        encoder = self._get_encoder()
        query_emb = encoder.encode(query, normalize_embeddings=True)
        
        # Find most similar examples (excluding anchor and query itself)
        similarities = []
        for i, ex in enumerate(self.pool):
            if ex.qid in used_qids:
                continue
            sim = np.dot(query_emb, ex.embedding)
            similarities.append((sim, i, ex))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: -x[0])
        
        # Select top-k, ensuring diversity (different difficulties)
        seen_difficulties = {anchor.difficulty}
        for sim, idx, ex in similarities:
            if len(selected) >= 1 + self.num_dynamic:
                break
            
            # Prefer diverse difficulties, but accept if no choice
            if ex.difficulty not in seen_difficulties or len(selected) >= self.num_dynamic:
                selected.append(ex)
                seen_difficulties.add(ex.difficulty)
        
        return selected
    
    def format_examples(self, examples: list[FewShotExample]) -> str:
        """Format selected examples into prompt text."""
        parts = []
        
        for i, ex in enumerate(examples, 1):
            docs_text = "\n\n".join(
                f"[{j+1}] {doc['text']}" 
                for j, doc in enumerate(ex.documents)
            )
            
            example_text = f"""EXAMPLE {i} ({ex.difficulty.upper()} query):
Query: "{ex.query}"

{docs_text}

{{"reasoning": "{ex.reasoning}", "ranking": {ex.ranking}}}"""
            parts.append(example_text)
        
        return "\n\n".join(parts)
    
    def get_few_shot_prompt(
        self,
        query: str,
        qid: str = None,
    ) -> str:
        """
        Get formatted few-shot examples for a query.
        
        Args:
            query: The current query text
            qid: Query ID (to exclude from selection)
        
        Returns:
            Formatted few-shot prompt text
        """
        exclude = {qid} if qid else set()
        examples = self.select_examples(query, exclude_qids=exclude)
        return self.format_examples(examples)


# Convenience function for quick usage
def get_dynamic_few_shot(query: str, qid: str = None) -> str:
    """Get dynamic few-shot prompt for a query."""
    selector = DynamicFewShotSelector()
    return selector.get_few_shot_prompt(query, qid)
