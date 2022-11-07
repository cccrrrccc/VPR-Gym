#include "RL_agent_util.h"
#include "manual_move_generator.h"

void create_move_generators(std::unique_ptr<MoveGenerator>& move_generator, std::unique_ptr<MoveGenerator>& move_generator2, const t_placer_opts& placer_opts, int move_lim) {
    if (placer_opts.RL_agent_placement == false) {
        if (placer_opts.place_algorithm.is_timing_driven()) {
            VTR_LOG("Using static probabilities for choosing each move type\n");
            VTR_LOG("Probability of Uniform_move : %f \n", placer_opts.place_static_move_prob[0]);
            VTR_LOG("Probability of Median_move : %f \n", placer_opts.place_static_move_prob[1]);
            VTR_LOG("Probability of Centroid_move : %f \n", placer_opts.place_static_move_prob[2]);
            VTR_LOG("Probability of Weighted_centroid_move : %f \n", placer_opts.place_static_move_prob[3]);
            VTR_LOG("Probability of Weighted_median_move : %f \n", placer_opts.place_static_move_prob[4]);
            VTR_LOG("Probability of Timing_feasible_region_move : %f \n", placer_opts.place_static_move_prob[5]);
            VTR_LOG("Probability of Critical_uniform_move : %f \n", placer_opts.place_static_move_prob[6]);
            move_generator = std::make_unique<StaticMoveGenerator>(placer_opts.place_static_move_prob);
            move_generator2 = std::make_unique<StaticMoveGenerator>(placer_opts.place_static_move_prob);
        } else { //Non-timing driven placement
            VTR_LOG("Using static probabilities for choosing each move type\n");
            VTR_LOG("Probability of Uniform_move : %f \n", placer_opts.place_static_notiming_move_prob[0]);
            VTR_LOG("Probability of Median_move : %f \n", placer_opts.place_static_notiming_move_prob[1]);
            VTR_LOG("Probability of Centroid_move : %f \n", placer_opts.place_static_notiming_move_prob[2]);
            move_generator = std::make_unique<StaticMoveGenerator>(placer_opts.place_static_notiming_move_prob);
            move_generator2 = std::make_unique<StaticMoveGenerator>(placer_opts.place_static_notiming_move_prob);
        }
    } else { //RL based placement
        /* For the non timing driven placecment: the agent has a single state  *
         *     - Available actions are (Uniform / Median / Centroid)           *
         *                                                                     *
         * For the timing driven placement: the agent has two states           *
         *     - 1st state: includes 4 actions (Uniform / Median / Centroid /  *
         *                  WeightedCentroid)                                  *
         *     - 2nd state: includes 7 actions (Uniform / Median / Centroid /  *
         *                  WeightedCentroid / WeightedMedian / Feasible       *
         *                  Region / CriticalUniform)                          *
         *                                                                     *
         *      This state is activated late in the anneale and in the Quench  */

        if (placer_opts.place_agent_algorithm == E_GREEDY) {
            VTR_LOG("Using simple RL 'Epsilon Greedy agent' for choosing move types\n");
            std::unique_ptr<EpsilonGreedyAgent> karmed_bandit_agent1, karmed_bandit_agent2;
            if (placer_opts.place_algorithm.is_timing_driven()) {
                //agent's 1st state
                karmed_bandit_agent1 = std::make_unique<EpsilonGreedyAgent>(NUM_PL_1ST_STATE_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent1->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent1);
                //agent's 2nd state
                karmed_bandit_agent2 = std::make_unique<EpsilonGreedyAgent>(NUM_PL_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent2->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator2 = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent2);
            } else {
                //agent's 1st state
                karmed_bandit_agent1 = std::make_unique<EpsilonGreedyAgent>(NUM_PL_NONTIMING_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent1->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent1);
                //agent's 2nd state
                karmed_bandit_agent2 = std::make_unique<EpsilonGreedyAgent>(NUM_PL_NONTIMING_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent2->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator2 = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent2);
            }
        }
        else if (placer_opts.place_agent_algorithm == E_DECAY) {
            VTR_LOG("Using simple RL 'Epsilon Decay agent' for choosing move types\n");
            std::unique_ptr<EpsilonDecayAgent> karmed_bandit_agent1, karmed_bandit_agent2;
            if (placer_opts.place_algorithm.is_timing_driven()) {
                //agent's 1st state
                karmed_bandit_agent1 = std::make_unique<EpsilonDecayAgent>(NUM_PL_1ST_STATE_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent1->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent1);
                //agent's 2nd state
                karmed_bandit_agent2 = std::make_unique<EpsilonDecayAgent>(NUM_PL_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent2->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator2 = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent2);
            } else {
                //agent's 1st state
                karmed_bandit_agent1 = std::make_unique<EpsilonDecayAgent>(NUM_PL_NONTIMING_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent1->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent1);
                //agent's 2nd state
                karmed_bandit_agent2 = std::make_unique<EpsilonDecayAgent>(NUM_PL_NONTIMING_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent2->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator2 = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent2);
            }
        }
        else if (placer_opts.place_agent_algorithm == UCB) {
            VTR_LOG("Using simple RL 'UCB agent' for choosing move types\n");
            std::unique_ptr<UCBAgent> karmed_bandit_agent1, karmed_bandit_agent2;
            if (placer_opts.place_algorithm.is_timing_driven()) {
                //agent's 1st state
                karmed_bandit_agent1 = std::make_unique<UCBAgent>(NUM_PL_1ST_STATE_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent1->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent1);
                //agent's 2nd state
                karmed_bandit_agent2 = std::make_unique<UCBAgent>(NUM_PL_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent2->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator2 = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent2);
            } else {
                //agent's 1st state
                karmed_bandit_agent1 = std::make_unique<UCBAgent>(NUM_PL_NONTIMING_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent1->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent1);
                //agent's 2nd state
                karmed_bandit_agent2 = std::make_unique<UCBAgent>(NUM_PL_NONTIMING_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent2->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator2 = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent2);
            }
        }
        else if (placer_opts.place_agent_algorithm == UCB_1) {
            VTR_LOG("Using simple RL 'UCB-tuned agent' for choosing move types\n");
            std::unique_ptr<UCB1_Agent> karmed_bandit_agent1, karmed_bandit_agent2;
            if (placer_opts.place_algorithm.is_timing_driven()) {
                //agent's 1st state
                karmed_bandit_agent1 = std::make_unique<UCB1_Agent>(NUM_PL_1ST_STATE_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent1->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent1);
                //agent's 2nd state
                karmed_bandit_agent2 = std::make_unique<UCB1_Agent>(NUM_PL_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent2->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator2 = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent2);
            } else {
                //agent's 1st state
                karmed_bandit_agent1 = std::make_unique<UCB1_Agent>(NUM_PL_NONTIMING_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent1->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent1);
                //agent's 2nd state
                karmed_bandit_agent2 = std::make_unique<UCB1_Agent>(NUM_PL_NONTIMING_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent2->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator2 = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent2);
            }
        }
        else if (placer_opts.place_agent_algorithm == EXP3) {
            VTR_LOG("Using simple RL 'EXP3' for choosing move types\n");
            std::unique_ptr<EXP3Agent> karmed_bandit_agent1, karmed_bandit_agent2;
            if (placer_opts.place_algorithm.is_timing_driven()) {
                //agent's 1st state
                karmed_bandit_agent1 = std::make_unique<EXP3Agent>(NUM_PL_1ST_STATE_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent1->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent1);
                //agent's 2nd state
                karmed_bandit_agent2 = std::make_unique<EXP3Agent>(NUM_PL_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent2->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator2 = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent2);
            } else {
                //agent's 1st state
                karmed_bandit_agent1 = std::make_unique<EXP3Agent>(NUM_PL_NONTIMING_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent1->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent1);
                //agent's 2nd state
                karmed_bandit_agent2 = std::make_unique<EXP3Agent>(NUM_PL_NONTIMING_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent2->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator2 = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent2);
            }
        }
        else if (placer_opts.place_agent_algorithm == UCBC) {
            VTR_LOG("Using simple RL 'UCBC' for choosing move types\n");
            std::unique_ptr<UCBCAgent> karmed_bandit_agent1, karmed_bandit_agent2;
            if (placer_opts.place_algorithm.is_timing_driven()) {
                //agent's 1st state
                karmed_bandit_agent1 = std::make_unique<UCBCAgent>(NUM_PL_1ST_STATE_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent1->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent1);
                //agent's 2nd state
                karmed_bandit_agent2 = std::make_unique<UCBCAgent>(NUM_PL_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent2->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator2 = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent2);
            } else {
                //agent's 1st state
                karmed_bandit_agent1 = std::make_unique<UCBCAgent>(NUM_PL_NONTIMING_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent1->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent1);
                //agent's 2nd state
                karmed_bandit_agent2 = std::make_unique<UCBCAgent>(NUM_PL_NONTIMING_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent2->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator2 = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent2);
            }
        }
        else if (placer_opts.place_agent_algorithm == MOSS) {
            VTR_LOG("Using simple RL 'UCB agent' for choosing move types\n");
            std::unique_ptr<MOSSAgent> karmed_bandit_agent1, karmed_bandit_agent2;
            if (placer_opts.place_algorithm.is_timing_driven()) {
                //agent's 1st state
                karmed_bandit_agent1 = std::make_unique<MOSSAgent>(NUM_PL_1ST_STATE_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent1->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent1);
                //agent's 2nd state
                karmed_bandit_agent2 = std::make_unique<MOSSAgent>(NUM_PL_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent2->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator2 = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent2);
            } else {
                //agent's 1st state
                karmed_bandit_agent1 = std::make_unique<MOSSAgent>(NUM_PL_NONTIMING_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent1->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent1);
                //agent's 2nd state
                karmed_bandit_agent2 = std::make_unique<MOSSAgent>(NUM_PL_NONTIMING_MOVE_TYPES, placer_opts.place_agent_epsilon);
                karmed_bandit_agent2->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator2 = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent2);
            }
        }
        else {
            VTR_LOG("Using simple RL 'Softmax agent' for choosing move types\n");
            std::unique_ptr<SoftmaxAgent> karmed_bandit_agent1, karmed_bandit_agent2;

            if (placer_opts.place_algorithm.is_timing_driven()) {
                //agent's 1st state
                karmed_bandit_agent1 = std::make_unique<SoftmaxAgent>(NUM_PL_1ST_STATE_MOVE_TYPES);
                karmed_bandit_agent1->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent1);
                //agent's 2nd state
                karmed_bandit_agent2 = std::make_unique<SoftmaxAgent>(NUM_PL_MOVE_TYPES);
                karmed_bandit_agent2->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator2 = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent2);
            } else {
                //agent's 1st state
                karmed_bandit_agent1 = std::make_unique<SoftmaxAgent>(NUM_PL_NONTIMING_MOVE_TYPES);
                karmed_bandit_agent1->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent1);
                //agent's 2nd state
                karmed_bandit_agent2 = std::make_unique<SoftmaxAgent>(NUM_PL_NONTIMING_MOVE_TYPES);
                karmed_bandit_agent2->set_step(placer_opts.place_agent_gamma, move_lim);
                move_generator2 = std::make_unique<SimpleRLMoveGenerator>(karmed_bandit_agent2);
            }
        }
    }
}

void assign_current_move_generator(std::unique_ptr<MoveGenerator>& move_generator, std::unique_ptr<MoveGenerator>& move_generator2, e_agent_state agent_state, const t_placer_opts& placer_opts, bool in_quench, std::unique_ptr<MoveGenerator>& current_move_generator) {
    if (in_quench) {
        if (placer_opts.place_quench_algorithm.is_timing_driven() && placer_opts.place_agent_multistate == true)
            current_move_generator = std::move(move_generator2);
        else
            current_move_generator = std::move(move_generator);
    } else {
        if (agent_state == EARLY_IN_THE_ANNEAL || placer_opts.place_agent_multistate == false)
            current_move_generator = std::move(move_generator);
        else
            current_move_generator = std::move(move_generator2);
    }
}

void update_move_generator(std::unique_ptr<MoveGenerator>& move_generator, std::unique_ptr<MoveGenerator>& move_generator2, e_agent_state agent_state, const t_placer_opts& placer_opts, bool in_quench, std::unique_ptr<MoveGenerator>& current_move_generator) {
    if (in_quench) {
        if (placer_opts.place_quench_algorithm.is_timing_driven() && placer_opts.place_agent_multistate == true)
            move_generator2 = std::move(current_move_generator);
        else
            move_generator = std::move(current_move_generator);
    } else {
        if (agent_state == EARLY_IN_THE_ANNEAL || placer_opts.place_agent_multistate == false)
            move_generator = std::move(current_move_generator);
        else
            move_generator2 = std::move(current_move_generator);
    }
}

void create_gym_generator(std::unique_ptr<RLGymGenerator>& move_generator, const t_placer_opts& placer_opts, int move_lim, size_t num_actions) {
    move_generator = std::make_unique<RLGymGenerator>(num_actions, placer_opts, move_lim);
}

void delete_gym_generator(std::unique_ptr<RLGymGenerator>& move_generator) {
    move_generator.reset();
}