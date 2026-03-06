import numpy as np
import stim
import pymatching
from tqdm import tqdm
import time


def State_Encoding(d, num_layer, p_local = 0.001, p_trans = 0.001):
    circuit = stim.Circuit()
    
    data_qubit_per_row = d
    data_qubit_per_col = d
    

    data_qubits_list = []
    stabilizer_x_list = []
    stabilizer_z_list = []
    qubit_index_position = {}

    # append data qubit
    for col in range(data_qubit_per_row):
        index_shift = col
        for row in range(data_qubit_per_col):
            X = 2 * row + 1
            Y = 2 * col + 1
            index = 2 * (row + col * d) + 1 + index_shift
            circuit.append("QUBIT_COORDS", [index], [X, Y])
            data_qubits_list.append(index)
            qubit_index_position[index] = (X, Y)

    # append boundary z stabilizer ancilla qubit
    for col in range(data_qubit_per_row):
        index_shift = col
        # up z boundary - data qubit - right up 
        for row in [0]:
            if col % 2 == 1:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1
            x_z = X - 1
            y_z = Y - 1
            if y_z <= 0:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_z = index - 1
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            stabilizer_z_list.append(index_z)
            qubit_index_position[index_z] = (x_z, y_z)
        # down z boundary - data qubit - right down
        for row in [data_qubit_per_col - 1]:
            if col % 2 == 0:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1
            x_z = X + 1
            y_z = Y - 1
            if y_z <= 0:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_z = index + 1
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            stabilizer_z_list.append(index_z)
            qubit_index_position[index_z] = (x_z, y_z)

    # append boundary x stabilizer ancilla qubit
    for row in range(data_qubit_per_col):
        # right x boundary - data qubit - right up 
        for col in [0]:
            index_shift = col
            if row % 2 == 1:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1
            x_x = X + 1
            y_x = Y - 1
            if x_x >= data_qubit_per_col * 2:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_x = index + 1
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            stabilizer_x_list.append(index_x)
            qubit_index_position[index_x] = (x_x, y_x)
        # left x boundary - data qubit - right down
        for col in [data_qubit_per_row - 1]:
            index_shift = col
            if row % 2 == 0:
                continue
            X = 2 * row + 1
            Y = 2 * col + 1
            x_x = X + 1
            y_x = Y + 1
            index = 2 * (row + col * d) + 1 + index_shift
            index_x = index + 1 + (2 * d + 1)
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            stabilizer_x_list.append(index_x)
            qubit_index_position[index_x] = (x_x, y_x)

    # append z stabilizer (general case)
    for row in range(data_qubit_per_col - 1):
        for col in range(1, data_qubit_per_row):
            index_shift = col
            X = 2 * row + 1
            Y = 2 * col + 1
            x_z = X + 1
            y_z = Y - 1
            if ((x_z + y_z) // 2) % 2 == 1:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_z = index + 1
            circuit.append("QUBIT_COORDS", [index_z], [x_z, y_z])
            stabilizer_z_list.append(index_z)
            qubit_index_position[index_z] = (x_z, y_z)

    # append x stabilizer (general case)
    for row in range(data_qubit_per_col - 1):
        for col in range(1, data_qubit_per_row):
            index_shift = col
            X = 2 * row + 1
            Y = 2 * col + 1
            x_x = X + 1
            y_x = Y - 1
            if ((x_x + y_x) // 2) % 2 == 0:
                continue
            index = 2 * (row + col * d) + 1 + index_shift
            index_x = index + 1
            circuit.append("QUBIT_COORDS", [index_x], [x_x, y_x])
            stabilizer_x_list.append(index_x)
            qubit_index_position[index_x] = (x_x, y_x)

    # circuit.append("R", [index_x], [x_x, y_x])
    # print the circuit


    qubits_all_list = data_qubits_list + stabilizer_x_list + stabilizer_z_list
    stabilizer_list = stabilizer_x_list + stabilizer_z_list
    
    circuit.append("R", qubits_all_list)
    circuit.append('X_ERROR', qubits_all_list, p_local)
    circuit.append("TICK")

    def syndrome_extraction(circuit):
        circuit.append("H", stabilizer_x_list)
        circuit.append("TICK")
        # 1 - Syndrome Extraction
        CX_list1 = [] 
        for i in stabilizer_x_list:
            if i + 1 in data_qubits_list:
                CX_list1.extend([i, i + 1])
        for i in data_qubits_list:
            if i - 1 in stabilizer_z_list:
                CX_list1.extend([i, i - 1])
                
        single1 = list(set(qubits_all_list) - set(CX_list1))

        circuit.append('CX', CX_list1)
        circuit.append('DEPOLARIZE2', CX_list1, p_local)
        circuit.append('DEPOLARIZE1', single1, p_local)
        circuit.append("TICK")

        # 2 - Syndrome Extraction
        CX_list2 = [] 
        for i in stabilizer_x_list:
            if i - 1 in data_qubits_list:
                CX_list2.extend([i, i - 1])
        for i in data_qubits_list:
            if i + 2 * d in stabilizer_z_list:
                CX_list2.extend([i, i + 2 * d])
                
        single2 = list(set(qubits_all_list) - set(CX_list2))

        circuit.append('CX', CX_list2)
        circuit.append('DEPOLARIZE2', CX_list2, p_local)
        circuit.append('DEPOLARIZE1', single2, p_local)
        circuit.append("TICK")

        # 3 - Syndrome Extraction
        CX_list3 = [] 
        for i in stabilizer_x_list:
            if i - 2 * d in data_qubits_list:
                CX_list3.extend([i, i - 2 * d])
        for i in data_qubits_list:
            if i + 1 in stabilizer_z_list:
                CX_list3.extend([i, i + 1])
                
        single3 = list(set(qubits_all_list) - set(CX_list3))

        circuit.append('CX', CX_list3)
        circuit.append('DEPOLARIZE2', CX_list3, p_local)
        circuit.append('DEPOLARIZE1', single3, p_local)
        circuit.append("TICK")

        # 4 - Syndrome Extraction
        CX_list4 = [] 
        for i in stabilizer_x_list:
            if i - (2 * d + 2) in data_qubits_list:
                CX_list4.extend([i, i - (2 * d + 2)])
        for i in data_qubits_list:
            if i + (2 * d + 2) in stabilizer_z_list:
                CX_list4.extend([i, i + (2 * d + 2)])
                
        single4 = list(set(qubits_all_list) - set(CX_list4))

        circuit.append('CX', CX_list4)
        circuit.append('DEPOLARIZE2', CX_list4, p_local)
        circuit.append('DEPOLARIZE1', single4, p_local)
        circuit.append("TICK")

        circuit.append("H", stabilizer_x_list)
        circuit.append("TICK")
        
        circuit.append('X_ERROR', stabilizer_list, p_local)
        circuit.append('MR', stabilizer_list)
        circuit.append('X_ERROR', stabilizer_list, p_local)
        
        return circuit

    circuit = syndrome_extraction(circuit)


    # first layer : only z detectors, no x detectors
    detector_shift = 0
    for i in stabilizer_list[::-1]:
        detector_shift -= 1
        if i in stabilizer_z_list:
            X, Y = qubit_index_position[i]
            circuit.append("DETECTOR",[stim.target_rec(detector_shift)], [X, Y, 0])

    # Add transmission error once
    circuit.append('DEPOLARIZE1', data_qubits_list, p_trans)
    
    # Add repeat block
    repeated_block = stim.Circuit()
    # repeated_block.append('DEPOLARIZE1', data_qubits_list + data_qubits_list, p_local)
    repeated_block.append("TICK")
    repeated_block = syndrome_extraction(repeated_block)
    repeated_block.append("SHIFT_COORDS", [], [0, 0, 1])

    # mid layer : all x detectors and z detectors
    detector_shift = 0
    for i in stabilizer_list[::-1]:
        detector_shift -= 1
        X, Y = qubit_index_position[i]
        repeated_block.append("DETECTOR",[stim.target_rec(detector_shift), stim.target_rec(detector_shift - len(stabilizer_list))], [X, Y, 0])

    circuit.append(stim.CircuitRepeatBlock(num_layer, repeated_block))

    # measure all data qubits
    # circuit.append('X_ERROR', data_qubits_list, p_local)

    circuit.append('M', data_qubits_list)


    detector_list = []
    detector_list += stabilizer_list
    detector_list += data_qubits_list
    detector_shift_list = [(-len(detector_list) + i) for i in range(len(detector_list))]

    # for memory circuit, perform "perfect" measurement of z stabilizers
    for i in range(len(detector_list)):
        if detector_list[i] in stabilizer_z_list:   
            target_rec = []
            target_rec.append(stim.target_rec(detector_shift_list[i]))
            # 1 stabilizer - "4" data qubits
            for j in range(len(detector_list)):
                if detector_list[j] not in data_qubits_list:
                    continue
                if detector_list[j] == detector_list[i] - 1:
                    target_rec.append(stim.target_rec(detector_shift_list[j]))
                if detector_list[j] == detector_list[i] + 1:
                    target_rec.append(stim.target_rec(detector_shift_list[j]))
                if detector_list[j] == detector_list[i] - 2 * d - 2:
                    target_rec.append(stim.target_rec(detector_shift_list[j]))
                if detector_list[j] == detector_list[i] - 2 * d:
                    target_rec.append(stim.target_rec(detector_shift_list[j]))
            
            X, Y = qubit_index_position[detector_list[i]]
            circuit.append("DETECTOR",target_rec, [X, Y, 1])
                
    obs_targets = [stim.target_rec(-j - 1) for j in range(d)]
    circuit.append(stim.CircuitInstruction('OBSERVABLE_INCLUDE', obs_targets, [0]))

    return circuit
