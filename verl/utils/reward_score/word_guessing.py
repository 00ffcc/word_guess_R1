import re

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    
    # 删去solution_str中所有<response> </response>标签内的内容（包括标签）

    solution_str = re.sub(r"<\|im_start\|>system.*?<\|im_end\|>", "", solution_str, flags=re.DOTALL)
    solution_str = re.sub(r"<\|im_start\|>user.*?<\|im_end\|>", "", solution_str, flags=re.DOTALL)
    solution_str = re.sub(r'<response>(.*?)</response>', '', solution_str, flags=re.DOTALL)


    # 用正则表达式提取出所有<query>标签中的内容</query>
    solution_str = re.findall(r'<query>(.*?)</query>', solution_str, re.DOTALL)

    solution_str = [s.upper() for s in solution_str]

    if ground_truth['target'] in solution_str:
        return 1
        # return len(solution_str) * (-0.01) + 1
    return 0