import re

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """
    用正则表达式提取出所有<query>标签中的内容</query>
    """
    solution_str = re.findall(r'<query>(.*?)</query>', solution_str, re.DOTALL)

    if ground_truth in solution_str:
        return len(solution_str) * (-0.01) + 1
    return 0