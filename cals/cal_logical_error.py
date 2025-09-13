import numpy as np
import stim
import pymatching
from tqdm import tqdm
import time
from ldpc.bplsd_decoder import BpLsdDecoder
import beliefmatching
from qldpc import decoder, codes

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from circuits.surface import *
from circuits.steane import *
from circuits.DEJMPS import *

def cal_DEJMPS_detection(depth, p_local=0.001, p_trans=0.001, shots = 1000000):
    print(f"depth = {depth}, local error = {p_local}, trans error = {p_trans}")
    basis_num = 4 ** (2 ** depth - 1)
    fidelity_basis = "ZX"
    check_successes_all = 0
    successes_all = 0
    

    circuit = epr_distillation(n = depth, p_local = p_local, p_trans=p_trans, fidelity_basis=fidelity_basis)
    check_successes, successes = simulate_detection_circuit(circuit, shots=shots)
    check_successes_all += check_successes
    successes_all += successes
    
    if check_successes == 0:
        print("No syndrome check success, protocal failed!")
    return check_successes_all / shots, 1 - successes_all / check_successes_all

def cal_surface_correction(d = 5, num_layer = 5, p_local = 0.001, p_trans = 0.01, shots = 3000000, protocol_type="encoding"):
    if protocol_type == "encoding":
        # QECC Protocal ZZ measurement -> Send to Bob 
        circuit = EPR_QECC_Protocal_Circuit(d = d, num_layer = num_layer, p_local = p_local, p_trans = p_trans)
        
    elif protocol_type == "distillation":
        # LOCC Protocal EPR pairs -> Send to Bob 
        circuit = EPR_LOCC_Protocal_Circuit(d = d, num_layer = num_layer, p_local = p_local, p_trans = p_trans)
    
    elif protocol_type == "QECC":
        circuit = Direct_QECC_Circuit(d = d, num_layer = num_layer, p_local = p_local, p_trans = p_trans)
    
    else:
        raise ValueError("Invalid protocol type.")

    sampler = circuit.compile_detector_sampler()
    syndrome_matrix, actual_observables = sampler.sample(shots=shots, separate_observables=True)
    model = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(model)
    predicted_observables = matching.decode_batch(syndrome_matrix)
    return (predicted_observables ^ actual_observables).sum() / shots
    
def cal_surface_detection(d = 5, num_layer = 5, p_local = 0.001, p_trans = 0.01, shots = 10000, protocol_type="encoding"):
    if protocol_type == "encoding":
        # QECC Protocal ZZ measurement -> Send to Bob 
        circuit = EPR_QECC_Protocal_Circuit(d = d, num_layer = num_layer, p_local = p_local, p_trans = p_trans)
        
    elif protocol_type == "distillation":
        # LOCC Protocal EPR pairs -> Send to Bob 
        circuit = EPR_LOCC_Protocal_Circuit(d = d, num_layer = num_layer, p_local = p_local, p_trans = p_trans)
    
    elif protocol_type == "QECC":
        circuit = Direct_QECC_Circuit(d, num_layer, p_local = p_local, p_trans = p_trans)
    
    else:
        raise ValueError("Invalid protocol type.")

    sampler = circuit.compile_detector_sampler()
    syndrome_matrix, actual_observables = sampler.sample(shots=shots, separate_observables=True)
    # calculate 1 in syndrome 
    syndrome_weights = syndrome_matrix.sum(axis=1)
    threshold = 0

    valid_indices = syndrome_weights <= threshold 
    filtered_syndrome_matrix = syndrome_matrix[valid_indices]
    filtered_actual_observables = actual_observables[valid_indices]

    logical_error_count = filtered_actual_observables.sum()
    filtered_n = len(filtered_actual_observables)
    return  filtered_n / shots, logical_error_count / filtered_n if filtered_n > 0 else 1

def cal_2g_surface_correction(d = 5, p_local = 0.001, p_trans = 0.1, n = 10000):
    circuit_qecc = EE_Surface_Transversal_CNOT_Teleportation(d = d, p_local = p_local, p_trans = p_trans)
    sampler = circuit_qecc.compile_detector_sampler()
    model = circuit_qecc.detector_error_model(decompose_errors=True)
    syndrome_matrix, actual_observables = sampler.sample(shots=n, separate_observables=True)
    matching = pymatching.Matching.from_detector_error_model(model)
    predicted_observables = matching.decode_batch(syndrome_matrix)
    return (predicted_observables ^ actual_observables).sum() / n

def cal_2g_surface_detection(d = 5, p_local = 0.001, p_trans = 0.1, n = 10000):
    circuit_qecc = EE_Surface_Transversal_CNOT_Teleportation(d = d, p_local = p_local, p_trans = p_trans)
    sampler = circuit_qecc.compile_detector_sampler()
    model = circuit_qecc.detector_error_model(decompose_errors=True)
    syndrome_matrix, actual_observables = sampler.sample(shots=n, separate_observables=True)
    syndrome_weights = syndrome_matrix.sum(axis=1)
    threshold = 0
    valid_indices = syndrome_weights <= threshold 
    filtered_syndrome_matrix = syndrome_matrix[valid_indices]
    filtered_actual_observables = actual_observables[valid_indices]
    logical_error_count = filtered_actual_observables.sum()
    filtered_n = len(filtered_actual_observables)
    return  filtered_n / n, logical_error_count / filtered_n if filtered_n > 0 else 1
    
def cal_steane_correction(p_local = 0.001, p_trans = 0.1, shots=10000, protocol_type=1):
        # circuit = stim.Circuit.from_file("test.stim")
    protocol_circuits = {
        1: ED_circuit,
        2: EE_circuit,
        3: QECC_circuit_Plus
    }
    Code = codes.SteaneCode()
    
    circuit = protocol_circuits[protocol_type](Code=Code, p_local=p_local, p_trans=p_trans)
    sampler = circuit.compile_detector_sampler()
    syndrome_matrix, actual_observables = sampler.sample(shots=shots, separate_observables=True)


    print(syndrome_matrix.shape)

    model = circuit.detector_error_model()
    matrices = beliefmatching.detector_error_model_to_check_matrices(model, allow_undecomposed_hyperedges=True)

    bplsd = BpLsdDecoder(
            matrices.check_matrix,
            channel_probs=matrices.priors,
            max_iter=30,
            osd_order=14,
            bp_method='product_sum',
            osd_method='lsd_cs',
            schedule='serial'
    )

    successful_decodes = 0
    OUR_Decoder = bplsd
    for i in tqdm(range(len(syndrome_matrix))):
        corr = OUR_Decoder.decode(syndrome_matrix[i])
        if ((matrices.observables_matrix @ corr) % 2 == actual_observables[i]).all():
            successful_decodes += 1

    return (shots - successful_decodes) / shots

def cal_2g_steane_correction(p_local = 0.001, p_trans = 0.1, shots=10000):
    Code = codes.SteaneCode()
    circuit = EE_circuit_Transversal_CNOT_Teleportation_Aligned(Code=Code, p_local=p_local, p_trans=p_trans)
    sampler = circuit.compile_detector_sampler()
    syndrome_matrix, actual_observables = sampler.sample(shots=shots, separate_observables=True)

    print(syndrome_matrix.shape)

    model = circuit.detector_error_model()
    matrices = beliefmatching.detector_error_model_to_check_matrices(model, allow_undecomposed_hyperedges=True)

    bplsd = BpLsdDecoder(
            matrices.check_matrix,
            channel_probs=matrices.priors,
            max_iter=30,
            osd_order=14,
            bp_method='product_sum',
            osd_method='lsd_cs',
            schedule='serial'
    )

    successful_decodes = 0
    OUR_Decoder = bplsd
    for i in tqdm(range(len(syndrome_matrix))):
        corr = OUR_Decoder.decode(syndrome_matrix[i])
        if ((matrices.observables_matrix @ corr) % 2 == actual_observables[i]).all():
            successful_decodes += 1

    return (shots - successful_decodes) / shots

def cal_steane_detection(p_local = 0.001, p_trans = 0.1, shots=10000, protocol_type=1):
        # circuit = stim.Circuit.from_file("test.stim")
    protocol_circuits = {
        1: ED_circuit,
        2: EE_circuit,
        4: QECC_circuit_Plus
    }
    Code = codes.SteaneCode()
    
    circuit = protocol_circuits[protocol_type](Code=Code, p_local=p_local, p_trans=p_trans)
    sampler = circuit.compile_detector_sampler()
    syndrome_matrix, actual_observables = sampler.sample(shots=shots, separate_observables=True)
    # calculate 1 in syndrome 
    syndrome_weights = syndrome_matrix.sum(axis=1)
    threshold = 0

    valid_indices = syndrome_weights <= threshold 
    filtered_syndrome_matrix = syndrome_matrix[valid_indices]
    filtered_actual_observables = actual_observables[valid_indices]

    logical_error_count = filtered_actual_observables.sum()
    filtered_n = len(filtered_actual_observables)
    return  filtered_n / shots, logical_error_count / filtered_n if filtered_n > 0 else 1
