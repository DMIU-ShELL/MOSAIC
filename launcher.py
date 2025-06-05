#######################################################################
# Copyright (C) 2025 Saptarshi Nath, Christos Peridis,                #
# Eseoghene Benjamin, Andrea Soltoggio                                #
# Licensed under the Apache License, Version 2.0                      #
# http://www.apache.org/licenses/LICENSE-2.0                          #
#######################################################################

import subprocess
import shlex
import time
import os
import argparse

commands_ctgraph_four_distributions = [
    # IMG SEED 1 AGENTS
    # MIG 1 (0-7)
    ['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', "python run_mctgraph.py 0 29500 -l -d 0.0 --exp_id='ct28_a0_d1'"],
    ['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', "python run_mctgraph.py 1 29501 -l -d 0.0 --exp_id='ct28_a1_d1'"],
    # MIG 2 (0-8)
    ['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', "python run_mctgraph.py 2 29502 -l -d 0.0 --exp_id='ct28_a2_d1'"],
    ['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', "python run_mctgraph.py 3 29503 -l -d 0.0 --exp_id='ct28_a3_d1'"],
    # MIG 3 (0-9)
    ['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', "python run_mctgraph.py 4 29504 -l -d 0.0 --exp_id='ct28_a4_d1'"],
    ['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', "python run_mctgraph.py 5 29505 -l -d 0.0 --exp_id='ct28_a5_d1'"],
    # MIG 4 (0-10)
    ['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', "python run_mctgraph.py 6 29506 -l -d 0.0 --exp_id='ct28_a6_d1'"],


    # IMG SEED 2 AGENTS
    ['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', "python run_mctgraph.py 7 29507 -l -d 0.0 --exp_id='ct28_a7_d2'"],
    # MIG 5 (0-11)
    ['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', "python run_mctgraph.py 8 29508 -l -d 0.0 --exp_id='ct28_a8_d2'"],
    ['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', "python run_mctgraph.py 9 29509 -l -d 0.0 --exp_id='ct28_a9_d2'"],
    # MIG 6 (0-12)
    ['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', "python run_mctgraph.py 10 29510 -l -d 0.0 --exp_id='ct28_a13_d2'"],
    ['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', "python run_mctgraph.py 11 29511 -l -d 0.0 --exp_id='ct28_a14_d2'"],
    # MIG 7 (0-13)
    ['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', "python run_mctgraph.py 12 29512 -l -d 0.0 --exp_id='ct28_a15_d2'"],
    ['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', "python run_mctgraph.py 13 29513 -l -d 0.0 --exp_id='ct28_a16_d2'"],


    # IMG SEED 3 AGENTS
    # MIG 8 (1-7)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', "python run_mctgraph.py 14 29514 -l -d 0.0 --exp_id='ct28_a20_d3'"],
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', "python run_mctgraph.py 15 29515 -l -d 0.0 --exp_id='ct28_a21_d3'"],
    # MIG 9 (1-8)
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', "python run_mctgraph.py 16 29516 -l -d 0.0 --exp_id='ct28_a22_d3'"],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', "python run_mctgraph.py 17 29517 -l -d 0.0 --exp_id='ct28_a23_d3'"],
    # MIG 10 (1-9)
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', "python run_mctgraph.py 18 29518 -l -d 0.0 --exp_id='ct28_a24_d3'"],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', "python run_mctgraph.py 19 29519 -l -d 0.0 --exp_id='ct28_a25_d3'"],
    # MIG 11 (1-10)
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', "python run_mctgraph.py 20 29520 -l -d 0.0 --exp_id='ct28_a26_d3'"],


    # IMG SEED 4 AGENTS
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', "python run_mctgraph.py 21 29521 -l -d 0.0 --exp_id='ct28_a30_d4'"],
    # MIG 12 (1-11)
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', "python run_mctgraph.py 22 29522 -l -d 0.0 --exp_id='ct28_a31_d4'"],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', "python run_mctgraph.py 23 29523 -l -d 0.0 --exp_id='ct28_a32_d4'"],
    # MIG 13 (1-12)
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', "python run_mctgraph.py 24 29524 -l -d 0.0 --exp_id='ct28_a33_d4'"],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', "python run_mctgraph.py 25 29525 -l -d 0.0 --exp_id='ct28_a34_d4'"],
    # MIG 14 (1-13)
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', "python run_mctgraph.py 26 29526 -l -d 0.0 --exp_id='ct28_a35_d4'"],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', "python run_mctgraph.py 27 29527 -l -d 0.0 --exp_id='ct28_a36_d4'"]
]

commands_ctgraph_four_distributions_baseline = [
    
    # IMG SEED 1 AGENTS
    # MIG 1 (0-7)
    ['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', "python run_mctgraph_baseline.py 0 29500 -l -d 0.0 --exp_id='ct28_a0_d1'"],
    ['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', "python run_mctgraph_baseline.py 1 29501 -l -d 0.0 --exp_id='ct28_a1_d1'"],
    # MIG 2 (0-8)
    ['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', "python run_mctgraph_baseline.py 2 29502 -l -d 0.0 --exp_id='ct28_a2_d1'"],
    ['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', "python run_mctgraph_baseline.py 3 29503 -l -d 0.0 --exp_id='ct28_a3_d1'"],
    # MIG 3 (0-9)
    ['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', "python run_mctgraph_baseline.py 4 29504 -l -d 0.0 --exp_id='ct28_a4_d1'"],
    ['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', "python run_mctgraph_baseline.py 5 29505 -l -d 0.0 --exp_id='ct28_a5_d1'"],
    # MIG 4 (0-10)
    ['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', "python run_mctgraph_baseline.py 6 29506 -l -d 0.0 --exp_id='ct28_a6_d1'"],


    # IMG SEED 2 AGENTS
    ['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', "python run_mctgraph_baseline.py 7 29507 -l -d 0.0 --exp_id='ct28_a7_d2'"],
    # MIG 5 (0-11)
    ['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', "python run_mctgraph_baseline.py 8 29508 -l -d 0.0 --exp_id='ct28_a8_d2'"],
    ['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', "python run_mctgraph_baseline.py 9 29509 -l -d 0.0 --exp_id='ct28_a9_d2'"],
    # MIG 6 (0-12)
    ['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', "python run_mctgraph_baseline.py 10 29510 -l -d 0.0 --exp_id='ct28_a13_d2'"],
    ['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', "python run_mctgraph_baseline.py 11 29511 -l -d 0.0 --exp_id='ct28_a14_d2'"],
    # MIG 7 (0-13)
    ['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', "python run_mctgraph_baseline.py 12 29512 -l -d 0.0 --exp_id='ct28_a15_d2'"],
    ['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', "python run_mctgraph_baseline.py 13 29513 -l -d 0.0 --exp_id='ct28_a16_d2'"],


    # IMG SEED 3 AGENTS
    # MIG 8 (1-7)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', "python run_mctgraph_baseline.py 14 29514 -l -d 0.0 --exp_id='ct28_a20_d3'"],
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', "python run_mctgraph_baseline.py 15 29515 -l -d 0.0 --exp_id='ct28_a21_d3'"],
    # MIG 9 (1-8)
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', "python run_mctgraph_baseline.py 16 29516 -l -d 0.0 --exp_id='ct28_a22_d3'"],
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', "python run_mctgraph_baseline.py 17 29517 -l -d 0.0 --exp_id='ct28_a23_d3'"],
    # MIG 10 (1-9)
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', "python run_mctgraph_baseline.py 18 29518 -l -d 0.0 --exp_id='ct28_a24_d3'"],
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', "python run_mctgraph_baseline.py 19 29519 -l -d 0.0 --exp_id='ct28_a25_d3'"],
    # MIG 11 (1-10)
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', "python run_mctgraph_baseline.py 20 29520 -l -d 0.0 --exp_id='ct28_a26_d3'"],


    # IMG SEED 4 AGENTS
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', "python run_mctgraph_baseline.py 21 29521 -l -d 0.0 --exp_id='ct28_a30_d4'"],
    # MIG 12 (1-11)
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', "python run_mctgraph_baseline.py 22 29522 -l -d 0.0 --exp_id='ct28_a31_d4'"],
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', "python run_mctgraph_baseline.py 23 29523 -l -d 0.0 --exp_id='ct28_a32_d4'"],
    # MIG 13 (1-12)
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', "python run_mctgraph_baseline.py 24 29524 -l -d 0.0 --exp_id='ct28_a33_d4'"],
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', "python run_mctgraph_baseline.py 25 29525 -l -d 0.0 --exp_id='ct28_a34_d4'"],
    # MIG 14 (1-13)
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', "python run_mctgraph_baseline.py 26 29526 -l -d 0.0 --exp_id='ct28_a35_d4'"],
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', "python run_mctgraph_baseline.py 27 29527 -l -d 0.0 --exp_id='ct28_a36_d4'"]
]

commands_minigrid = [
    # MIG 1 (0-7)
    ['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', "python run_minigrid.py 0 29500 -l -d 0.0 --exp_id='mg_agent1'"],

    # MIG 2 (0-8)
    ['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', "python run_minigrid.py 1 29501 -l -d 0.0 --exp_id='mg_agent2'"],

    # MIG 3 (0-9)
    ['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', "python run_minigrid.py 2 29502 -l -d 0.0 --exp_id='mg_agent3'"],

    # MIG 4 (0-10)
    ['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', "python run_minigrid.py 3 29503 -l -d 0.0 --exp_id='mg_agent4'"],

    # MIG 5 (0-11)
    ['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', "python run_minigrid.py 4 29504 -l -d 0.0 --exp_id='mg_agent5'"],

    # MIG 6 (0-12)
    ['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', "python run_minigrid.py 5 29505 -l -d 0.0 --exp_id='mg_agent6'"],

    # MIG 7 (0-13)
    ['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', "python run_minigrid.py 6 29506 -l -d 0.0 --exp_id='mg_agent7'"],

    # MIG 8 (1-7)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', "python run_minigrid.py 7 29507 -l -d 0.0 --exp_id='mg_agent8'"],

    # MIG 9 (1-8)
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', "python run_minigrid.py 8 29508 -l -d 0.0 --exp_id='mg_agent9'"],

    # MIG 10 (1-9)
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', "python run_minigrid.py 9 29509 -l -d 0.0 --exp_id='mg_agent10'"],

    # MIG 11 (1-10)
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', "python run_minigrid.py 10 29510 -l -d 0.0 --exp_id='mg_agent11'"],

    # MIG 12 (1-11)
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', "python run_minigrid.py 11 29511 -l -d 0.0 --exp_id='mg_agent12'"],

    # MIG 13 (1-12)
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', "python run_minigrid.py 12 29512 -l -d 0.0 --exp_id='mg_agent13'"],

    # MIG 14 (1-13)
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', "python run_minigrid.py 13 29513 -l -d 0.0 --exp_id='mg_agent14'"]
]

commands_minihack = [
    # MIG 1 (0-7)
    ['MIG-c3ce33ce-ced8-5961-bb87-2b40eb100277', "python run_minihack.py 0 29500 -l -d 0.0 --exp_id='minihack_agent1'"],

    # MIG 2 (0-8)
    ['MIG-280489c4-1d98-5b07-b4f6-2fc85fc874fa', "python run_minihack.py 1 29501 -l -d 0.0 --exp_id='minihack_agent2'"],

    # MIG 3 (0-9)
    ['MIG-c432df19-0894-5232-ac1c-9a3440fc267e', "python run_minihack.py 2 29502 -l -d 0.0 --exp_id='minihack_agent3'"],

    # MIG 4 (0-10)
    ['MIG-e8f61a95-352a-56cc-b95d-0c35fc14e8bf', "python run_minihack.py 3 29503 -l -d 0.0 --exp_id='minihack_agent4'"],

    # MIG 5 (0-11)
    ['MIG-35ecef79-db2e-590b-9e8c-2c07c787008e', "python run_minihack.py 4 29504 -l -d 0.0 --exp_id='minihack_agent5'"],

    # MIG 6 (0-12)
    ['MIG-76cd8dd7-7703-5581-8ac5-a7ee81a402a0', "python run_minihack.py 5 29505 -l -d 0.0 --exp_id='minihack_agent6'"],

    # MIG 7 (0-13)
    ['MIG-b35e1a68-f7a4-5ef9-b34a-1abf6d1f8c2e', "python run_minihack.py 6 29506 -l -d 0.0 --exp_id='minihack_agent7'"],



    # MIG 8 (1-7)
    ['MIG-2d5b6364-fc42-587b-97c6-ee316a82e2f3', "python run_minihack.py 7 29507 -l -d 0.0 --exp_id='minihack_agent8'"],

    # MIG 9 (1-8)
    ['MIG-4590f80d-be70-58e4-af75-eeb950255d4a', "python run_minihack.py 8 29508 -l -d 0.0 --exp_id='minihack_agent9'"],
    
    # MIG 10 (1-9)
    ['MIG-e76a2a9b-9867-5f8a-b145-d857cd5ed8e2', "python run_minihack.py 9 29509 -l -d 0.0 --exp_id='minihack_agent10'"],

    # MIG 11 (1-10)
    ['MIG-2593b912-5975-58e9-bc3d-495311cee807', "python run_minihack.py 10 29510 -l -d 0.0 --exp_id='minihack_agent11'"],

    # MIG 12 (1-11)
    ['MIG-51069529-f343-59c6-bac7-a75648296e7b', "python run_minihack.py 11 29511 -l -d 0.0 --exp_id='minihack_agent12'"],

    # MIG 13 (1-12)
    ['MIG-187573d8-7df7-5e5f-87b2-c8b8f73c54e7', "python run_minihack.py 12 29512 -l -d 0.0 --exp_id='minihack_agent13'"],

    # MIG 14 (1-13)
    ['MIG-3045e3dd-28b6-5ee8-96b5-60a085c9fcf1', "python run_minihack.py 13 29513 -l -d 0.0 --exp_id='minihack_agent14'"],
]

commands_minigrid_overlapped = [
    # MIG 1 (0-7)
    ['MIG-c8cbc779-6499-5d1f-86e7-ad789e047331', "python run_minigrid.py 0 29500 -l -d 0.0 --exp_id='mg_agent1'"],
    ['MIG-c8cbc779-6499-5d1f-86e7-ad789e047331', "python run_minigrid.py 1 29501 -l -d 0.0 --exp_id='mg_agent2'"],

    # MIG 2 (0-8)
    ['MIG-f6c4eed6-1edf-5515-8e0b-0a5521e81cbb', "python run_minigrid.py 2 29502 -l -d 0.0 --exp_id='mg_agent3'"],
    ['MIG-f6c4eed6-1edf-5515-8e0b-0a5521e81cbb', "python run_minigrid.py 3 29503 -l -d 0.0 --exp_id='mg_agent4'"],

    # MIG 3 (0-9)
    ['MIG-ae777ac5-da68-5606-8581-1878288224dc', "python run_minigrid.py 4 29504 -l -d 0.0 --exp_id='mg_agent5'"],
    ['MIG-ae777ac5-da68-5606-8581-1878288224dc', "python run_minigrid.py 5 29505 -l -d 0.0 --exp_id='mg_agent6'"],

    # MIG 4 (0-10)
    ['MIG-b2a723b5-db7a-50d8-a267-70569dafe609', "python run_minigrid.py 6 29506 -l -d 0.0 --exp_id='mg_agent7'"],

    # MIG 5 (0-11)
    #['MIG-73000f9d-03d4-5dab-ad23-34b546569bdd', "python run_minigrid.py 4 29504 -l -d 0.0 --exp_id='mg_agent5'"],

    # MIG 6 (0-12)
    #['MIG-f86cf155-b80e-56da-92ed-03c78bf647c7', "python run_minigrid.py 5 29505 -l -d 0.0 --exp_id='mg_agent6'"],

    # MIG 7 (0-13)
    #['MIG-bfb3fc5a-8b5e-59c1-87e8-3468c995103d', "python run_minigrid.py 6 29506 -l -d 0.0 --exp_id='mg_agent7'"],

    # MIG 8 (1-7)
    ['MIG-f61216a8-2a31-500e-acf7-f0158fbf7ce3', "python run_minigrid.py 7 29507 -l -d 0.0 --exp_id='mg_agent8'"],
    ['MIG-f61216a8-2a31-500e-acf7-f0158fbf7ce3', "python run_minigrid.py 8 29508 -l -d 0.0 --exp_id='mg_agent9'"],

    # MIG 9 (1-8)
    ['MIG-bc91396f-1c2b-5319-8862-6f16d089ce5e', "python run_minigrid.py 9 29509 -l -d 0.0 --exp_id='mg_agent10'"],
    ['MIG-bc91396f-1c2b-5319-8862-6f16d089ce5e', "python run_minigrid.py 10 29510 -l -d 0.0 --exp_id='mg_agent11'"],

    # MIG 10 (1-9)
    ['MIG-13d6b3aa-c302-5ac2-9183-5494d22547e6', "python run_minigrid.py 11 29511 -l -d 0.0 --exp_id='mg_agent12'"],
    ['MIG-13d6b3aa-c302-5ac2-9183-5494d22547e6', "python run_minigrid.py 12 29512 -l -d 0.0 --exp_id='mg_agent13'"],

    # MIG 11 (1-10)
    ['MIG-f2d8b14c-d00c-5b5f-be98-bd1bff3bf371', "python run_minigrid.py 13 29513 -l -d 0.0 --exp_id='mg_agent14'"],

    # MIG 12 (1-11)
    #['MIG-62fabbf0-b8de-5040-b4db-93f62c477543', "python run_minigrid.py 11 29511 -l -d 0.0 --exp_id='mg_agent12'"],

    # MIG 13 (1-12)
    #['MIG-fcda8ac4-e82d-5259-8495-c777d8c95d74', "python run_minigrid.py 12 29512 -l -d 0.0 --exp_id='mg_agent13'"],

    # MIG 14 (1-13)
    #['MIG-f4aecf22-d8a0-50a9-b804-c65cb2d2ff7e', "python run_minigrid.py 13 29514 -l -e -d 0.0 --exp_id='mg_evaluation'"]
]

parser = argparse.ArgumentParser()
parser.add_argument('--env', help='indicate which experiment is being run for command selection', type=str, default='minigrid')
parser.add_argument('--exp', help='', type=str, default='')
args = parser.parse_args()
commands = None
if args.env == 'minigrid':
    commands = commands_minigrid
elif args.env == 'minihack':
    commands = commands_minihack
elif args.env =='mgo':
    commands = commands_minigrid_overlapped
elif args.env == 'ctfour':
    commands = commands_ctgraph_four_distributions
elif args.env == 'ctbaseline':
    commands = commands_ctgraph_four_distributions_baseline
else:
    raise ValueError(f'no commands have been setup for --exp {args.exp}')


env = dict(os.environ)


path_header = args.env
if len(args.exp) > 0:
    path_header = args.exp


# Run the commands in seperate terminals
processes = []
for command in commands:
    print(f"{command[0]}, {command[1]} -p {path_header}")
    if len(command[0]) != 0:
        env['CUDA_VISIBLE_DEVICES'] = command[0]
    process = subprocess.Popen(shlex.split(command[1] + f' -p {path_header}'), env=env)
    processes.append(process)
    #time.sleep(5)

for process in processes:
    stdout, stderr = process.communicate()