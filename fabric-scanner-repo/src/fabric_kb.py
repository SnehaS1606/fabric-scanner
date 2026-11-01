"""
fabric_kb.py
Structured knowledge base + question-intent routing for the RAG layer.

answer_from_kb(question, probs, class_names) -> (answer, route)
    route == "kb"      : answered from the knowledge base (fabric-property question)
    route == "visual"  : question is about how the sample looks -> send to the VLM as-is
    route == "open"    : no intent matched -> VLM with retrieved facts as context
"""

import re

import numpy as np

KB = {
    "Cotton": {
        "season": "Cotton suits spring and summer best, and works all year round with layering.",
        "about": "Cotton is a soft, breathable natural fibre.",
        "summer": "Yes, cotton is a good summer fabric because it is breathable and absorbent.",
        "winter": "Not on its own. Cotton is not very warm, so it needs layering in winter.",
        "durability": "Cotton is moderately durable, but it wrinkles and can shrink in hot washes.",
        "care": "Cotton is usually machine washable. Avoid very hot water to prevent shrinking.",
        "breathability": "Cotton is very breathable and absorbs sweat well.",
    },
    "Denim": {
        "season": "Denim suits autumn, winter and cooler spring days best. It is less comfortable in hot summer weather.",
        "about": "Denim is a heavy, tightly woven cotton twill.",
        "summer": "Not ideal. Denim is heavy and not very breathable, so it can feel hot in summer.",
        "winter": "Yes, denim suits cooler weather, though layer up on very cold days as it is not strongly insulating.",
        "durability": "Denim is very durable and hard wearing.",
        "care": "Wash denim cold and inside out. It dries slowly and fades over time.",
        "breathability": "Denim is not very breathable because it is a heavy, tightly woven fabric.",
    },
    "Silk": {
        "season": "Silk suits spring and summer best, and works as a light layer in cooler seasons.",
        "about": "Silk is a smooth, lustrous, lightweight natural fibre.",
        "summer": "Yes, silk is lightweight and comfortable in warm weather, though it shows sweat and water marks.",
        "winter": "Only as a light layer. Silk is not very insulating alone, but it works well under warmer clothes.",
        "durability": "Silk is strong for its weight but delicate. It can snag and weaken with sunlight and rough washing.",
        "care": "Hand wash silk gently in cool water or dry clean it. Avoid soaking and harsh detergent.",
        "breathability": "Silk is breathable and helps regulate body temperature.",
    },
    "Wool": {
        "season": "Wool suits autumn and winter best.",
        "about": "Wool is a warm, insulating natural fibre.",
        "summer": "Usually not. Wool is very warm and is too hot for most summer weather.",
        "winter": "Yes, wool is excellent for winter. It insulates well and stays warm even when damp.",
        "durability": "Wool is durable and resilient, but it can pill and shrink if washed hot.",
        "care": "Hand wash wool or use a gentle cold cycle, and dry it flat to avoid shrinking.",
        "breathability": "Wool is breathable and wicks moisture, but it is very warm.",
    },
    "Polyester": {
        "season": "Polyester suits cooler seasons and active wear best, and can feel sticky in summer heat.",
        "about": "Polyester is a durable synthetic fibre.",
        "summer": "Not ideal. Polyester is less breathable, so it can feel hot and sticky in summer.",
        "winter": "It depends. Polyester fleece is warm, but thin polyester fabric is not very insulating.",
        "durability": "Polyester is very durable, resists wrinkles and holds its shape and colour.",
        "care": "Polyester is easy to care for. Machine wash it and it dries quickly.",
        "breathability": "Polyester is not very breathable and can trap heat and sweat.",
    },
    "Leather": {
        "season": "Leather suits autumn and winter best.",
        "about": "Leather is a durable, wind-resistant material.",
        "summer": "Not ideal. Leather is not breathable, so it feels hot in warm weather.",
        "winter": "Yes, leather blocks wind and holds warmth, so it suits autumn and winter.",
        "durability": "Leather is very durable and can last for many years with care.",
        "care": "Wipe leather with a damp cloth, condition it regularly, and keep it away from soaking and direct heat.",
        "breathability": "Leather is not very breathable.",
    },
}


# Additional per-fabric facts (uses, age suitability, skin comfort, stretch/wrinkles, water, cost, environment)
KB_EXTRA = {
    "Cotton": {
        "uses": "Cotton is used for everyday clothing such as T-shirts, shirts, dresses and underwear, and for bedsheets and towels.",
        "age": "Cotton is gentle and commonly used for babies, children and adults of all ages.",
        "skin": "Cotton is soft and rarely irritates skin, so it suits sensitive skin.",
        "stretch": "Cotton wrinkles easily and has little stretch unless it is blended with elastane.",
        "waterproof": "No. Cotton absorbs water and dries slowly.",
        "cost": "Cotton is usually affordable, though price varies a lot with quality.",
        "eco": "Cotton is natural and biodegradable, but conventional cotton farming uses a lot of water. Organic cotton uses less.",
    },
    "Denim": {
        "uses": "Denim is used for jeans, jackets, skirts, bags and workwear.",
        "age": "Denim is worn by all ages, but its stiff, heavy feel is less comfortable for babies. Soft or stretch denim is easier for young children.",
        "skin": "New denim can feel rough and may rub sensitive skin until it softens with washing.",
        "stretch": "Traditional denim has little stretch, though many jeans blend in elastane. Denim resists wrinkles well.",
        "waterproof": "No. Denim is not waterproof and dries slowly once soaked.",
        "cost": "Price varies widely, from cheap jeans to premium denim.",
        "eco": "Denim is cotton based, and making and dyeing it can use a lot of water and chemicals.",
    },
    "Silk": {
        "uses": "Silk is used for sarees, scarves, blouses, shirts, nightwear, ties and bedding.",
        "age": "Silk is worn by all ages, but it is delicate and needs gentle care, which makes it less practical for babies and small children.",
        "skin": "Silk is smooth and generally gentle on skin.",
        "stretch": "Silk wrinkles easily and has little stretch.",
        "waterproof": "No. Water can leave marks on silk.",
        "cost": "Silk is usually one of the more expensive fabrics.",
        "eco": "Silk is natural and biodegradable, but it comes from silkworms, which some people avoid.",
    },
    "Wool": {
        "uses": "Wool is used for sweaters, coats, blankets, socks, suits and shawls.",
        "age": "Wool is worn at all ages, but coarse wool can feel itchy on babies' skin. Fine merino wool is softer.",
        "skin": "Coarse wool can feel itchy on sensitive skin. Fine merino wool is softer.",
        "stretch": "Wool has natural elasticity and resists wrinkles.",
        "waterproof": "Not waterproof, but wool repels light rain and stays warm when damp.",
        "cost": "Wool is usually mid to high priced, and fine merino costs more.",
        "eco": "Wool is natural, renewable and biodegradable.",
    },
    "Polyester": {
        "uses": "Polyester is used for sportswear, jackets, dresses, uniforms, curtains, bags and fleece.",
        "age": "Polyester is worn by all ages, but it traps heat and sweat, so it is less comfortable for babies and in hot weather.",
        "skin": "Polyester can feel sweaty or irritate sensitive skin because it does not breathe well.",
        "stretch": "Polyester resists wrinkles and holds its shape. It stretches only when blended with elastane.",
        "waterproof": "Polyester is water resistant and dries quickly, but it is not fully waterproof unless it is coated.",
        "cost": "Polyester is usually inexpensive.",
        "eco": "Polyester is made from petroleum, is not biodegradable, and sheds microplastics when washed.",
    },
    "Leather": {
        "uses": "Leather is used for jackets, shoes, belts, bags, wallets and upholstery.",
        "age": "Leather is worn mostly by teenagers and adults. It is heavy and stiff for babies and small children.",
        "skin": "Leather is usually fine on skin, though tanning chemicals can irritate a few people.",
        "stretch": "Leather stretches slightly and moulds to the body over time rather than wrinkling like cloth.",
        "waterproof": "Leather is water resistant but not waterproof, and water can stain it unless it is treated.",
        "cost": "Real leather is usually expensive. Imitation leather is cheaper.",
        "eco": "Leather lasts a long time, but it is an animal product and tanning can be polluting.",
    },
}
for _fabric, _facts in KB_EXTRA.items():
    KB[_fabric].update(_facts)


KB_ORIGIN = {
    "Cotton": "Cotton is grown mainly in India, China, the United States and Brazil.",
    "Denim": "Denim is made mainly in China, Bangladesh, Pakistan, India and Turkey.",
    "Silk": "Most silk comes from China, followed by India.",
    "Wool": "Wool comes mainly from Australia, China and New Zealand.",
    "Polyester": "Polyester is manufactured mostly in China, India and other parts of Asia.",
    "Leather": "Leather is produced mainly in China, India, Italy and Brazil.",
}
for _fabric, _text in KB_ORIGIN.items():
    KB[_fabric]["origin"] = _text + " Which countries supply it to you depends on where you live."

AUTHENTICITY = {
    "Leather": "I can't reliably tell real leather from imitation from a photo alone. "
               "Check the label, the smell, and whether the edges and grain look natural.",
}
AUTHENTICITY_DEFAULT = ("I can't verify the exact fibre content from a photo alone. "
                        "Please check the care label.")

INTENT_KEYWORDS = {
    "authenticity": ["real", "fake", "genuine", "authentic", "pure", "imitation", "faux", "original"],
    "summer": ["summer", "hot weather", "heat", "sunny", "humid", "warm weather", "hot"],
    "winter": ["winter", "cold", "freez", "chilly", "keep me warm", "warm enough", "warm"],
    "season": ["season", "which weather", "what weather", "best time", "when can i wear", "when should i wear"],
    "durability": ["durab", "last long", "long lasting", "strong", "wear out", "tough", "sturdy"],
    "care": ["wash", "clean", "care", "iron", "dry", "shrink", "maintain", "laundry"],
    "breathability": ["breath", "sweat", "airy", "ventilat"],
    "about": ["what is", "what fabric", "what material", "what kind", "which fabric",
              "which material", "tell me about", "identify", "what type"],
}

# Word-boundary patterns for the extra intents (avoids matches like "cold" -> "old" or "because" -> "use")
REGEX_INTENTS = {
    "uses": [r"\bfunctional\w*", r"\buses\b", r"\bused (for|in|to)\b", r"\bpurposes?\b", r"\bapplications?\b",
             r"\bwhat can i (make|do)\b", r"\bwhat is it for\b"],
    "age": [r"\bages?\b", r"\bold(er)?\b", r"\bbab(y|ies)\b", r"\bkids?\b", r"\bchild(ren)?\b", r"\btoddlers?\b",
            r"\binfants?\b", r"\belderly\b", r"\badults?\b", r"\bteen\w*", r"\bwho can wear\b"],
    "skin": [r"\bskin\b", r"\bitch\w*", r"\bsensitive\b", r"\ballerg\w*", r"\birritat\w*", r"\brash\b",
             r"\bsoft(ness)?\b", r"\bcomfort\w*", r"\bscratch\w*", r"\brough\b"],
    "stretch": [r"\bstretch\w*", r"\belastic\w*", r"\bwrinkl\w*", r"\bcreas\w*"],
    "waterproof": [r"\bwaterproof\b", r"\bwater[- ]?resistan\w*", r"\brain\b", r"\bwet\b", r"\bmonsoon\b", r"\bwater\b"],
    "cost": [r"\bprice\w*", r"\bcosts?\b", r"\bexpensive\b", r"\bcheap\w*", r"\bafford\w*", r"\bbudget\b"],
    "blend": [r"\bblend\w*", r"\bmix(ed|ture)?\b", r"\bfusion\b", r"\bcombination\b", r"\bcomposite\b",
              r"\btwo (materials|fabrics)\b", r"\bmore than one\b"],
    "origin": [r"\bimport\w*", r"\bexport\w*", r"\bcountr\w*", r"\borigin\w*", r"\bgrown\b", r"\bmade in\b",
               r"\bproduced\b", r"\bproduction\b", r"\bmanufactur\w*", r"\bwhere (is|does|do|are)\b.*\b(from|come|made|grown)\b"],
    "eco": [r"\beco\w*", r"\bsustainab\w*", r"\bbiodegrad\w*", r"\benvironment\w*", r"\borganic\b"],
}

VISUAL_KEYWORDS = ["colour", "color", "pattern", "stripe", "texture", "shiny", "glossy",
                   "look like", "looks like", "describe", "see", "visible", "thread", "weave"]


def detect_intents(question):
    q = question.lower()
    found = []
    if any(k in q for k in INTENT_KEYWORDS["authenticity"]):
        return ["authenticity"]
    q_no_warm_weather = q.replace("warm weather", "")
    for intent in ["summer", "winter", "season", "durability", "care", "breathability", "about"]:
        text = q if intent != "winter" else q_no_warm_weather
        if any(k in text for k in INTENT_KEYWORDS[intent]):
            found.append(intent)
    if "season" in found and ("summer" in found or "winter" in found):
        found.remove("season")      # "summer season" is a summer question, not a best-season question
    extra = [name for name, pats in REGEX_INTENTS.items() if any(re.search(p, q) for p in pats)]
    if extra:
        found = [i for i in found if i != "about"] + [e for e in extra if e not in found]
    return found


def facts_text(fabric):
    return " ".join(KB[fabric].values())


def retrieve_facts(probs, class_names, second_min=0.2):
    order = np.argsort(-probs)
    parts = [facts_text(class_names[order[0]])]
    if probs[order[1]] >= second_min:
        parts.append(facts_text(class_names[order[1]]))
    return " ".join(parts)


def answer_from_kb(question, probs, class_names):
    order = np.argsort(-probs)
    top = class_names[order[0]]
    intents = detect_intents(question)

    # Questions that name a specific fabric ("is this cotton?", "does this look like real leather?")
    mentioned = [c for c in class_names if c.lower() in question.lower()]
    if mentioned and intents in ([], ["authenticity"]):
        top_p = probs[order[0]] * 100
        if mentioned[0] != top:
            m_p = probs[class_names.index(mentioned[0])] * 100
            return (f"The classifier thinks this is {top.lower()} ({top_p:.0f}%), "
                    f"not {mentioned[0].lower()} ({m_p:.0f}%)."), "kb"
        if intents == []:
            return f"Yes, the classifier thinks this is {top.lower()} with {top_p:.0f}% confidence.", "kb"

    if intents == ["authenticity"]:
        return AUTHENTICITY.get(top, AUTHENTICITY_DEFAULT), "kb"

    if "blend" in intents:
        if len(intents) == 1:
            second, p1, p2 = class_names[order[1]], probs[order[0]] * 100, probs[order[1]] * 100
            if probs[order[1]] >= 0.2:
                return (f"It could be. The classifier gives {top.lower()} {p1:.0f}% and {second.lower()} {p2:.0f}%, which can "
                        f"happen with a blend. But it is trained to name one fabric, so check the care label for the exact mix."), "kb"
            return (f"The classifier is fairly sure it is {top.lower()} ({p1:.0f}%). It cannot detect blends from a photo, "
                    f"so check the care label for the exact mix."), "kb"
        intents = [i for i in intents if i != "blend"]

    if intents:
        answer = " ".join(KB[top][i] for i in intents[:2])
        return answer, "kb"

    q = question.lower()
    if any(k in q for k in VISUAL_KEYWORDS):
        return None, "visual"
    return None, "open"
