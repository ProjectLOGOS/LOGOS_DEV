    def infer_trinity(self,
                      keywords: List[str],
                      weights: Optional[List[float]] = None,
                      enforce_coherence: bool = True
                     ) -> Dict[str, Any]:
        if not keywords:
            raise ValueError("Provide at least one keyword.")
        kws = [k.lower() for k in keywords]
        wts = weights or [1.0]*len(kws)

        # Accumulate weighted E, G, T
        e = g = t = sw = 0.0
        sources = []
        for kw, w in zip(kws, wts):
            entry = self.priors.get(kw)
            if entry:
                e += entry["E"] * w
                g += entry["G"] * w
                t += entry["T"] * w
                sw += w
                sources.append(kw)
        if sw == 0:
            raise ValueError("No matching priors.")
        e, g, t = e/sw, g/sw, t/sw

        # Compute ideal goodness and raw coherence C
        ideal_g = e * t
        c_raw = (g / ideal_g) if ideal_g > 0 else 0.0
        c_raw = min(1.0, c_raw)

        # Optionally enforce coherence by bumping G up to ideal
        adjusted = False
        if enforce_coherence and g < ideal_g:
            g = ideal_g
            adjusted = True

        # Complex coordinate remains unchanged
        z = complex(e * t, g)

        return {
            # ETGC tuple now includes C = raw coherence
            "trinity": (e, g, t, c_raw),
            "complex_coordinate": z,
            "sources": sources,
            "coherence": {
                "raw":   c_raw,
                "adjusted": adjusted,
                "ideal_goodness": ideal_g
            }
        }
