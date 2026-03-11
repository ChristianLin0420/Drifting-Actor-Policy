#!/bin/bash
# Run smoke test for all small/medium datasets
# Finished datasets: taco_play, stanford_hydra, cmu_stretch, aloha, bc_z, utaustin_mutex nyu_franka dexora bridgev2
# kuka berkeley_fanuc cmu_play_fusion jaco_play austin_buds austin_sirius columbia_pusht nyu_door

# Need to check: dexwild, rlbench droid droid behavior1k_t0000-t0049 austin_sailor

for ds in austin_sirius \
          columbia_pusht nyu_door; do
    echo ""
    echo "========================================"
    echo "  Testing: $ds"
    echo "========================================"
    bash test.sh all "$ds"
    if [ $? -ne 0 ]; then
        echo "❌ FAILED: $ds"
        exit 1
    fi
    echo "✅ PASSED: $ds"
done

echo ""
echo "========================================"
echo "  All datasets passed!"
echo "========================================"
