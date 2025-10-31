import time
import random
import math
from datetime import datetime
import requests
from bs4 import BeautifulSoup

from translation_engine import translate
from bayesian_inferencer import BayesianTrinityInferencer
from burt_module import filter_and_score
from causal_inference import learn_structure
from modal_vector_space import load_second_order_anchors
from banach_generator import BanachGenerator
from principles import (
    sign_principle,
    bridge_principle,
    mind_principle,
    non_contradiction_principle
)


class TrinitarianAgent:
    """
    One of the three agents (Father, Son, Spirit) that explores the vector space:
      - picks nodes
      - scrapes data
      - applies principle‑specific logic
      - spawns new nodes
      - emits return trips when near Julia anchors
      - amalgamates buffered snippets every third visit
    """
    def __init__(self, name, axis_name, return_threshold=0.1):
        self.name             = name
        self.axis             = axis_name
        self._traverse_cnt    = 0
        self._snippet_buf     = []
        self._trail           = []
        self.return_threshold = return_threshold

    def process_vector_space(self, banach_nodes, julia_anchors):
        # 1) Pick an origin node at random
        origin = random.choice(banach_nodes)
        snippets = self.generate_snippets(origin)
        self._snippet_buf.extend(snippets)

        # 2) Prepare chain data with principle‑specific scoring
        data = self.generate_chain_data(origin, snippets, julia_anchors)

        # 3) Validate and spawn a new node if checks pass
        if self.etgc_check(data) and self.bayesian_validate(data):
            new_node = self.spawn_node(data)
            coord = new_node['final_coords']
        else:
            coord = origin['final_coords']

        # 4) Append visited coordinate to trail
        self._trail.append(coord)

        # 5) Every third visit, amalgamate buffered snippets
        self._traverse_cnt += 1
        if self._traverse_cnt % 3 == 0:
            self._amalgamate_and_spawn()

        # 6) Check proximity to Julia anchors and emit return trip if close
        for anchor in julia_anchors:
            anchor_coord = (anchor[0], anchor[1], coord[2])
            if math.dist(coord, anchor_coord) <= self.return_threshold:
                self._emit_return_trip(anchor_coord)
                break

    def generate_snippets(self, origin):
        payload = origin.get('payload')
        return DivineMind.search_web(str(payload))

    def generate_chain_data(self, origin, snippets, julias):
        data = {
            'origin':   origin,
            'snippets': snippets,
            'anchors':  julias[:3]
        }
        # Principle‑specific score added to metadata
        if self.axis == 'sign':
            data['score'] = sign_principle(origin.get('metrics', {}))
        elif self.axis == 'bridge':
            data['score'] = bridge_principle(origin.get('structural_p', 0.0))
        else:  # mind
            data['score'] = mind_principle(origin.get('metrics', {}))
        return data

    def etgc_check(self, data):
        """
        Run EGTC filter via burt_module.filter_and_score
        Returns True if data passes the confidence threshold.
        """
        valid = filter_and_score([data])
        return len(valid) > 0

    def bayesian_validate(self, data):
        """
        Use BayesianTrinityInferencer to infer on snippets and ensure no exceptions.
        """
        infer = BayesianTrinityInferencer()
        try:
            infer.infer(data.get('snippets', []))
            return True
        except Exception:
            return False

    def spawn_node(self, data):
        gen = BanachGenerator()
        return gen.generate_node(
            payload=data,
            agent=self.name.lower(),
            source='divine_mind',
            metrics=data['origin'].get('metrics', {}),
            structural_p=data['origin'].get('structural_p', 0.0),
            coherence=data['origin'].get('coherence', 0.5)
        )

    def _amalgamate_and_spawn(self):
        if not self._snippet_buf:
            return
        unique = list(dict.fromkeys(self._snippet_buf))
        payload = {'text': ' '.join(unique)}
        gen = BanachGenerator()
        gen.generate_node(
            payload=payload,
            agent=self.name.lower(),
            source='amalgamation_every_3rd',
            metrics={'connectivity_score':0.5,'sync_score':0.5,'covariance_score':0.5,'contradiction_score':0.5},
            structural_p=0.5,
            coherence=0.5
        )
        self._snippet_buf.clear()

    def _emit_return_trip(self, anchor_coord):
        origin = (0.0, 0.0, 0.0)
        path = self._trail + [anchor_coord] + [origin]
        for p, q in zip(path, path[1:]):
            print(f"{self.name} RETRACE: {p} → {q}")
        self._trail.clear()


class DivineMind:
    """
    Orchestrates TrinitarianAgents to explore the modal vector space:
      - loads anchors & nodes
      - runs web searches
      - drives agent processing loops
    """
    def __init__(self,
                 anchors_path='ONTOPROP_DICT.json',
                 nodes_log='nodes/banach_nodes_log.json'):
        self.julia_anchors = load_second_order_anchors(anchors_path)
        self.banach_gen    = BanachGenerator(log_path=nodes_log)
        self.banach_nodes  = self.banach_gen.nodes
        self.agents        = [
            TrinitarianAgent('Father', 'sign'),
            TrinitarianAgent('Son',    'bridge'),
            TrinitarianAgent('Spirit', 'mind')
        ]
        self.processing_interval = 5

    @staticmethod
    def search_web(query, num=5):
        """
        Scrape google for snippets related to `query`.
        """
        url = f"https://www.google.com/search?q={query}"
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, 'html.parser')
        return [el.get_text().strip() for el in soup.select(".result__snippet")[:num]]

    def activate_background_processing(self):
        """
        Main loop: each agent processes the vector space at intervals.
        """
        while True:
            for agent in self.agents:
                agent.process_vector_space(self.banach_nodes, self.julia_anchors)
            time.sleep(self.processing_interval)


if __name__ == '__main__':
    dm = DivineMind()
    print(f"Loaded {len(dm.banach_nodes)} existing nodes.")
    dm.activate_background_processing()
