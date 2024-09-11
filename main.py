from SystolicArray import SystolicArray
from SparseMatirx import SparseMatrix
from IndexBuffer import IndexBuffer
from SparseCompution import SparseCompution
import numpy as np


def evaluate(systolic_row,systolic_column,matrix_A,matrix_B,mode=0):
    normal_computation = SystolicArray(systolic_row,systolic_column)
    normal_computation.reset()
    c = normal_computation.compute(matrix_A.matrix,matrix_B)
    print("密集矩阵计算cycle:",normal_computation.cycle)
    buffer_width = [2,4,8,16,32]
    cycles = []
    for i in buffer_width:
        sparse_computation = SparseCompution(systolic_row,i,systolic_column)
        sparse_computation.systolic_array.reset()
        c = sparse_computation.compute(matrix_A,matrix_B,mode)
        cycle = sparse_computation.systolic_array.cycle
        cycles.append(cycle)
        print(f"脉动阵列为{systolic_row}x{systolic_column},buf宽度为{i},填充方式为{mode},cycle为{cycle}")


A = SparseMatrix(2048,2048,1/16)
B = np.random.randint(0,10,size=(2048,128))

evaluate(32,4,A,B,0)
evaluate(32,4,A,B,1)
evaluate(32,8,A,B,0)
evaluate(32,8,A,B,1)
evaluate(32,16,A,B,0)
evaluate(32,16,A,B,1)
evaluate(4,32,A,B,0)
evaluate(4,32,A,B,1)
evaluate(8,32,A,B,0)
evaluate(8,32,A,B,1)
evaluate(16,32,A,B,0)
evaluate(16,32,A,B,1)

