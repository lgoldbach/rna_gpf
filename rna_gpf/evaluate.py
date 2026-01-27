from rna_folding.utils import dotbracket_to_bp


def f1_score(ref: str, query: str, abstract_level: int = 0) -> float:
    """Compute F1 score for two RNA secondary structures

    Args:
        ref (str): Reference structure
        query (str): Query structure
        abstract_level (int, optional): Create sequences on abstract shape 
            level [0-5]. 0 is full dot-bracket notation, 5 is maximum 
            abstraction. Defaults to 0.

    Returns:
        bool: True if perfect match, False otherwise

    """
    ref_ = dotbracket_to_bp(ref)
    query_ = dotbracket_to_bp(query)

    tp = ref_.intersection(query_)
    fp = query_.difference(tp)
    fn = ref_.difference(tp)

    sens_denom = len(tp) + len(fn)
    if sens_denom == 0:
        sensitivity = 1
    else:
        sensitivity = len(tp) / sens_denom

    prec_denom = len(tp) + len(fp)
    if prec_denom == 0:
        precision = 0
    else:
        precision =  len(tp) / prec_denom

    if sensitivity and precision:  # avoid division by zero
        f1 = (2*sensitivity*precision) / (sensitivity+precision)
    else:
        f1 = 0

    return f1


def compare_db(A: str, B: str, abstract_level: int = 0) -> bool:
    """Compare two RNA secondary structrues in dot-bracket notation

    Args:
        A (str): First sequence
        B (str): Second sequence
        abstract_level (int, optional): Create sequences on abstract shape 
            level [0-5]. 0 is full dot-bracket notation, 5 is maximum 
            abstraction. Defaults to 0.

    Returns:
        bool: True if perfect match, False otherwise

    """
    ############
    ### Placeholder for RNA shape abstraction code
    ############

    if A == B:
        return True
    else:
        False

