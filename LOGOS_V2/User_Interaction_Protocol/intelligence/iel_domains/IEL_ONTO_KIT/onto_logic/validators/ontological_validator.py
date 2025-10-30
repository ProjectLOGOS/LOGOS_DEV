"""
ontological_validator.py

Ontological property validation module for Logos-AGI.
Ensures each property meets ETGC: Existence, Goodness, Truthfulness, Coherence.
"""
import json

class OntologicalValidator:
    def __init__(self, ontology_path):
        with open(ontology_path, 'r', encoding='utf-8') as f:
            self.raw = json.load(f)

    def validate_existence(self, name, meta):
        # Existence: c_value must parse to complex
        try:
            complex(meta.get('c_value', None))
            return True
        except Exception:
            return False

    def validate_goodness(self, name, meta):
        # Goodness: group field must be non-empty
        group = meta.get('group') or meta.get('synergy_group')
        return isinstance(group, str) and bool(group.strip())

    def validate_truthfulness(self, name, meta):
        # Truthfulness: no immediate self-referential links
        links = meta.get('recursive_links', [])
        return name not in links

    def validate_coherence(self, name, meta):
        # Coherence: at least one link, but not linking to all properties
        links = meta.get('recursive_links', [])
        total = len(self.raw)
        return 0 < len(links) < total

    def validate_property(self, name):
        meta = self.raw.get(name)
        if not meta:
            return False, {'error': 'Missing metadata'}
        results = {
            'existence': self.validate_existence(name, meta),
            'goodness': self.validate_goodness(name, meta),
            'truthfulness': self.validate_truthfulness(name, meta),
            'coherence': self.validate_coherence(name, meta)
        }
        return all(results.values()), results

    def validate_all(self):
        report = {}
        for name in self.raw:
            valid, detail = self.validate_property(name)
            report[name] = {'valid': valid, 'checks': detail}
        return report

# CLI support
if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'final_ontological_property_dict.json'
    validator = OntologicalValidator(path)
    report = validator.validate_all()
    valid = [n for n,r in report.items() if r['valid']]
    invalid = [n for n,r in report.items() if not r['valid']]
    print(f"Validated {len(valid)} properties, {len(invalid)} failed:")
    for n in invalid:
        print(f" - {n}: {report[n]['checks']}")
