def get_need_probs(need_probs_key):
    need_probs_dict = {
        'uniform': None,
        'kemp': {
            'MM': 21, 'MF': 16, 'MZy': 5, 'MBy': 7, 'M': 432, 'MZe': 5, 'MBe': 7,
            'FM': 21, 'FF': 16, 'FZy': 5, 'FBy': 7, 'F': 438, 'FZe': 5, 'FBe': 7,
            'Zy': 39, 'By': 51, 'Ze': 39, 'Be': 51, 'ZyD': 1, 'ZyS': 1.2,
            'ByD': 1, 'ByS': 1.2, 'D': 131, 'S': 146, 'ZeD': 1, 'ZeS': 1.2, 'BeD': 1, 'BeS': 1.2,
            'DD': 2, 'DS': 3, 'SD': 2, 'SS': 3
        },
        'dutch': {
            'MM': 1077.5, 'MF': 1081.5, 'MZy': 204.75, 'MBy': 254.5, 'M': 20706, 'MZe': 204.75, 'MBe': 254.5,
            'FM': 1077.5, 'FF': 1081.5, 'FZy': 204.75, 'FBy': 254.5, 'F': 22181, 'FZe': 204.75, 'FBe': 254.5,
            'Zy': 2697.5, 'By': 3328.5, 'Ze': 2697.5, 'Be': 3328.5, 'ZyD': 193.5, 'ZyS': 254.25,
            'ByD': 193.5, 'ByS': 254.25, 'D': 11958, 'S': 12490, 'ZeD': 193.5, 'ZeS': 254.25, 'BeD': 193.5, 'BeS': 254.25,
            'DD': 87, 'DS': 117, 'SD': 87, 'SS': 117
        }
    }

    return need_probs_dict.get(need_probs_key, None)