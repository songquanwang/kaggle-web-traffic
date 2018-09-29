__author__ = 'songquanwang'
from typing import List


def func(a: int, string: str) -> List[int or str]:
    list1 = []
    list1.append(a)
    list1.append(string)
    return list1


func(1200, 'aaa')
