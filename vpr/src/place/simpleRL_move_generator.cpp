#include "simpleRL_move_generator.h"
#include "globals.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>

#include "vtr_random.h"

/* File-scope routines */
//a scaled and clipped exponential function
static float scaled_clipped_exp(float x) { return std::exp(std::min(1000000 * x, float(3.0))); }

/*                                     *
 *                                     *
 *  RL move generator implementation   *
 *                                     *
 *                                     */
SimpleRLMoveGenerator::SimpleRLMoveGenerator(std::unique_ptr<SoftmaxAgent>& agent) {
    avail_moves.push_back(std::move(std::make_unique<UniformMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<MedianMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<CentroidMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<WeightedCentroidMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<WeightedMedianMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<CriticalUniformMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<FeasibleRegionMoveGenerator>()));

    karmed_bandit_agent = std::move(agent);
}

SimpleRLMoveGenerator::SimpleRLMoveGenerator(std::unique_ptr<EpsilonGreedyAgent>& agent) {
    avail_moves.push_back(std::move(std::make_unique<UniformMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<MedianMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<CentroidMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<WeightedCentroidMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<WeightedMedianMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<CriticalUniformMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<FeasibleRegionMoveGenerator>()));

    karmed_bandit_agent = std::move(agent);
}

SimpleRLMoveGenerator::SimpleRLMoveGenerator(std::unique_ptr<EpsilonDecayAgent>& agent) {
    avail_moves.push_back(std::move(std::make_unique<UniformMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<MedianMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<CentroidMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<WeightedCentroidMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<WeightedMedianMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<CriticalUniformMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<FeasibleRegionMoveGenerator>()));

    karmed_bandit_agent = std::move(agent);
}

SimpleRLMoveGenerator::SimpleRLMoveGenerator(std::unique_ptr<UCBAgent>& agent) {
    avail_moves.push_back(std::move(std::make_unique<UniformMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<MedianMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<CentroidMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<WeightedCentroidMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<WeightedMedianMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<CriticalUniformMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<FeasibleRegionMoveGenerator>()));

    karmed_bandit_agent = std::move(agent);
}

SimpleRLMoveGenerator::SimpleRLMoveGenerator(std::unique_ptr<UCB1_Agent>& agent) {
    avail_moves.push_back(std::move(std::make_unique<UniformMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<MedianMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<CentroidMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<WeightedCentroidMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<WeightedMedianMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<CriticalUniformMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<FeasibleRegionMoveGenerator>()));

    karmed_bandit_agent = std::move(agent);
}


SimpleRLMoveGenerator::SimpleRLMoveGenerator(std::unique_ptr<EXP3Agent>& agent) {
    avail_moves.push_back(std::move(std::make_unique<UniformMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<MedianMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<CentroidMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<WeightedCentroidMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<WeightedMedianMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<CriticalUniformMoveGenerator>()));
    avail_moves.push_back(std::move(std::make_unique<FeasibleRegionMoveGenerator>()));

    karmed_bandit_agent = std::move(agent);
}

e_create_move SimpleRLMoveGenerator::propose_move(t_pl_blocks_to_be_moved& blocks_affected, e_move_type& move_type, float rlim, const t_placer_opts& placer_opts, const PlacerCriticalities* criticalities) {
    move_type = (e_move_type)karmed_bandit_agent->propose_action();
    return avail_moves[(int)move_type]->propose_move(blocks_affected, move_type, rlim, placer_opts, criticalities);
}

void SimpleRLMoveGenerator::process_outcome(double reward, e_reward_function reward_fun) {
    karmed_bandit_agent->process_outcome(reward, reward_fun);
}

/*                                        *
 *                                        *
 *  K-Armed bandit agent implementation   *
 *                                        *
 *                                        */
void EpsilonGreedyAgent::process_outcome(double reward, e_reward_function reward_fun) {
    ++num_action_chosen_[last_action_];
    t_++;
    if (reward_fun == RUNTIME_AWARE || reward_fun == WL_BIASED_RUNTIME_AWARE)
        reward /= time_elapsed_[last_action_];
    //Determine step size
    float step = 0.;
    if (exp_alpha_ < 0.) {
        step = 1. / num_action_chosen_[last_action_]; //Incremental average
    } else if (exp_alpha_ <= 1) {
        step = exp_alpha_; //Exponentially wieghted average
    } else {
        VTR_ASSERT_MSG(false, "Invalid step size");
    }

    //Based on the outcome how much should our estimate of q change?
    float delta_q = step * (reward - q_[last_action_]);

    //Update the estimated value of the last action
    q_[last_action_] += delta_q;

    // Update the sum of reward
    //sum_reward_[last_action_] = sum_reward_[last_action_] + reward;
    // Step mode
    sum_reward_[last_action_] = sum_reward_[last_action_] + reward * step;
    sum_reward_square_[last_action_] = sum_reward_square_[last_action_] + std::pow(reward * step, 2);
    if (agent_info_file_) {
        fprintf(agent_info_file_, "%zu,", last_action_);
        fprintf(agent_info_file_, "%g,", reward);

        for (size_t i = 0; i < num_available_actions_; ++i) {
            fprintf(agent_info_file_, "%g,", q_[i]);
        }

        for (size_t i = 0; i < num_available_actions_; ++i) {
            fprintf(agent_info_file_, "%zu", num_action_chosen_[i]);
            if (i != num_available_actions_ - 1) {
                fprintf(agent_info_file_, ",");
            }
        }
        fprintf(agent_info_file_, "\n");
    }
}

void EpsilonDecayAgent::process_outcome(double reward, e_reward_function reward_fun) {
    ++num_action_chosen_[last_action_];
    t_++;
    if (reward_fun == RUNTIME_AWARE || reward_fun == WL_BIASED_RUNTIME_AWARE)
        reward /= time_elapsed_[last_action_];
    //Determine step size
    float step = 0.;
    if (exp_alpha_ < 0.) {
        step = 1. / num_action_chosen_[last_action_]; //Incremental average
    } else if (exp_alpha_ <= 1) {
        step = exp_alpha_; //Exponentially wieghted average
    } else {
        VTR_ASSERT_MSG(false, "Invalid step size");
    }

    //Based on the outcome how much should our estimate of q change?
    float delta_q = step * (reward - q_[last_action_]);

    //Update the estimated value of the last action
    q_[last_action_] += delta_q;

    // Update the sum of reward
    //sum_reward_[last_action_] = sum_reward_[last_action_] + reward;
    // Step mode
    sum_reward_[last_action_] = sum_reward_[last_action_] + reward * step;
    sum_reward_square_[last_action_] = sum_reward_square_[last_action_] + std::pow(reward * step, 2);
    if (agent_info_file_) {
        fprintf(agent_info_file_, "%zu,", last_action_);
        fprintf(agent_info_file_, "%g,", reward);

        for (size_t i = 0; i < num_available_actions_; ++i) {
            fprintf(agent_info_file_, "%g,", q_[i]);
        }

        for (size_t i = 0; i < num_available_actions_; ++i) {
            fprintf(agent_info_file_, "%zu", num_action_chosen_[i]);
            if (i != num_available_actions_ - 1) {
                fprintf(agent_info_file_, ",");
            }
        }
        fprintf(agent_info_file_, "\n");
    }
}
void SoftmaxAgent::process_outcome(double reward, e_reward_function reward_fun) {
    ++num_action_chosen_[last_action_];
    t_++;
    if (reward_fun == RUNTIME_AWARE || reward_fun == WL_BIASED_RUNTIME_AWARE)
        reward /= time_elapsed_[last_action_];
    //Determine step size
    float step = 0.;
    if (exp_alpha_ < 0.) {
        step = 1. / num_action_chosen_[last_action_]; //Incremental average
    } else if (exp_alpha_ <= 1) {
        step = exp_alpha_; //Exponentially wieghted average
    } else {
        VTR_ASSERT_MSG(false, "Invalid step size");
    }

    //Based on the outcome how much should our estimate of q change?
    float delta_q = step * (reward - q_[last_action_]);

    //Update the estimated value of the last action
    q_[last_action_] += delta_q;

    // Update the sum of reward
    //sum_reward_[last_action_] = sum_reward_[last_action_] + reward;
    // Step mode
    //sum_reward_[last_action_] = sum_reward_[last_action_] + reward * step;
    //sum_reward_square_[last_action_] = sum_reward_square_[last_action_] + std::pow(reward * step, 2);
    if (agent_info_file_) {
        fprintf(agent_info_file_, "%zu,", last_action_);
        fprintf(agent_info_file_, "%g,", reward);

        for (size_t i = 0; i < num_available_actions_; ++i) {
            fprintf(agent_info_file_, "%g,", q_[i]);
        }

        for (size_t i = 0; i < num_available_actions_; ++i) {
            fprintf(agent_info_file_, "%zu", num_action_chosen_[i]);
            if (i != num_available_actions_ - 1) {
                fprintf(agent_info_file_, ",");
            }
        }
        fprintf(agent_info_file_, "\n");
    }
}

/*                                  *
 *                                  *
 *  E-greedy agent implementation   *
 *                                  *
 *                                  */
EpsilonGreedyAgent::EpsilonGreedyAgent(size_t num_actions, float epsilon) {
    set_epsilon(epsilon);
    num_available_actions_ = num_actions;
    q_ = std::vector<float>(num_actions, 0.);
    num_action_chosen_ = std::vector<size_t>(num_actions, 0);
    sum_reward_ = std::vector<float>(num_actions, 0.);
    sum_reward_square_ = std::vector<float>(num_actions, 0.);
    cumm_epsilon_action_prob_ = std::vector<float>(num_actions, 1.0 / num_actions);
    if (agent_info_file_) {
        fprintf(agent_info_file_, "action,reward,");
        for (size_t i = 0; i < num_available_actions_; ++i) {
            fprintf(agent_info_file_, "q%zu,", i);
        }
        for (size_t i = 0; i < num_available_actions_; ++i) {
            fprintf(agent_info_file_, "n%zu,", i);
        }
        fprintf(agent_info_file_, "\n");
    }
    set_epsilon_action_prob();
}

EpsilonGreedyAgent::~EpsilonGreedyAgent() {
    if (agent_info_file_) vtr::fclose(agent_info_file_);
}

void EpsilonGreedyAgent::set_step(float gamma, int move_lim) {
    VTR_LOG("Setting egreedy step: %g\n", exp_alpha_);
    if (gamma < 0) {
        exp_alpha_ = -1; //Use sample average
    } else {
        //
        // For an exponentially wieghted average the fraction of total weight applied to
        // to moves which occured > K moves ago is:
        //
        //      gamma = (1 - alpha)^K
        //
        // If we treat K as the number of moves per temperature (move_lim) then gamma
        // is the fraction of weight applied to moves which occured > move_lim moves ago,
        // and given a target gamma we can explicitly calcualte the alpha step-size
        // required by the agent:
        //
        //     alpha = 1 - e^(log(gamma) / K)
        //
        float alpha = 1 - std::exp(std::log(gamma) / move_lim);
        exp_alpha_ = alpha;
    }
}

size_t EpsilonGreedyAgent::propose_action() {
    size_t action = 0;
    if (vtr::frand() < epsilon_) {
        /* Explore
         * With probability epsilon, choose randomly amongst all move types */
        float p = vtr::frand();
        auto itr = std::lower_bound(cumm_epsilon_action_prob_.begin(), cumm_epsilon_action_prob_.end(), p);
        action = itr - cumm_epsilon_action_prob_.begin();
    } else {
        /* Greedy (Exploit)
         * For probability 1-epsilon, choose the greedy action */
        auto itr = std::max_element(q_.begin(), q_.end());
        VTR_ASSERT(itr != q_.end());
        action = itr - q_.begin();
    }
    VTR_ASSERT(action < num_available_actions_);

    last_action_ = action;
    return action;
}

void EpsilonGreedyAgent::set_epsilon(float epsilon) {
    VTR_LOG("Setting egreedy epsilon: %g\n", epsilon);
    epsilon_ = epsilon;
}

void EpsilonGreedyAgent::set_epsilon_action_prob() {
    //initialize to equal probabilities
    std::vector<float> epsilon_prob(num_available_actions_, 1.0 / num_available_actions_);

    float accum = 0;
    for (size_t i = 0; i < num_available_actions_; ++i) {
        accum += epsilon_prob[i];
        cumm_epsilon_action_prob_[i] = accum;
    }
}

/*                                  *
 *                                  *
 *  Softmax agent implementation    *
 *                                  *
 *                                  */
SoftmaxAgent::SoftmaxAgent(size_t num_actions) {
    num_available_actions_ = num_actions;
    q_ = std::vector<float>(num_actions, 0.);
    exp_q_ = std::vector<float>(num_actions, 0.);
    num_action_chosen_ = std::vector<size_t>(num_actions, 0);
    action_prob_ = std::vector<float>(num_actions, 0.);
    sum_reward_ = std::vector<float>(num_actions, 0.);
    sum_reward_square_ = std::vector<float>(num_actions, 0.);
    cumm_action_prob_ = std::vector<float>(num_actions);
    if (agent_info_file_) {
        fprintf(agent_info_file_, "action,reward,");
        for (size_t i = 0; i < num_available_actions_; ++i) {
            fprintf(agent_info_file_, "q%zu,", i);
        }
        for (size_t i = 0; i < num_available_actions_; ++i) {
            fprintf(agent_info_file_, "n%zu,", i);
        }
        fprintf(agent_info_file_, "\n");
    }
    set_action_prob();
    //agent_info_file_ = vtr::fopen("agent_info.txt", "w");
}

SoftmaxAgent::~SoftmaxAgent() {
    if (agent_info_file_) vtr::fclose(agent_info_file_);
}

size_t SoftmaxAgent::propose_action() {
    set_action_prob();
    size_t action = 0;
    float p = vtr::frand();
    auto itr = std::lower_bound(cumm_action_prob_.begin(), cumm_action_prob_.end(), p);
    action = itr - cumm_action_prob_.begin();
    //To take care that the last element in cumm_action_prob_ might be less than 1 by a small value
    if (action == num_available_actions_)
        action = num_available_actions_ - 1;
    VTR_ASSERT(action < num_available_actions_);

    last_action_ = action;
    return action;
}

void SoftmaxAgent::set_action_prob() {
    //calculate the scaled and clipped explonential function for the estimated q value for each action
    std::transform(q_.begin(), q_.end(), exp_q_.begin(), scaled_clipped_exp);

    // calculate the sum of all scaled clipped expnential q values
    float sum_q = accumulate(exp_q_.begin(), exp_q_.end(), 0.0);

    if (sum_q == 0.0) { //action probabilities need to be initialized with equal values
        std::fill(action_prob_.begin(), action_prob_.end(), 1.0 / num_available_actions_);
    } else {
        // calculate the probability of each action as the ratio of scaled_clipped_exp(action(i))/sum(scaled_clipped_exponentials)
        for (size_t i = 0; i < num_available_actions_; ++i) {
            action_prob_[i] = exp_q_[i] / sum_q;
        }
    }

    // normalize all the action probabilities to guarantee the sum(all actyion probs) = 1
    float sum_prob = std::accumulate(action_prob_.begin(), action_prob_.end(), 0.0);
    std::transform(action_prob_.begin(), action_prob_.end(), action_prob_.begin(),
                   bind2nd(std::plus<float>(), (1.0 - sum_prob) / num_available_actions_));

    //calulcate the accumulative action probability of each action
    // e.g. if we have 5 actions with equal probability of 0.2, the cumm_action_prob will be {0.2,0.4,0.6,0.8,1.0}
    float accum = 0;
    for (size_t i = 0; i < num_available_actions_; ++i) {
        accum += action_prob_[i];
        cumm_action_prob_[i] = accum;
    }
}

void SoftmaxAgent::set_step(float gamma, int move_lim) {
    if (gamma < 0) {
        exp_alpha_ = -1; //Use sample average
    } else {
        //
        // For an exponentially weighted average the fraction of total weight applied
        // to moves which occured > K moves ago is:
        //
        //      gamma = (1 - alpha)^K
        //
        // If we treat K as the number of moves per temperature (move_lim) then gamma
        // is the fraction of weight applied to moves which occured > move_lim moves ago,
        // and given a target gamma we can explicitly calcualte the alpha step-size
        // required by the agent:
        //
        //     alpha = 1 - e^(log(gamma) / K)
        //
        float alpha = 1 - std::exp(std::log(gamma) / move_lim);
        exp_alpha_ = alpha;
    }
}

/*                                  *
 *                                  *
 *  E-decay agent implementation    *
 *                                  *
 *                                  */
EpsilonDecayAgent::EpsilonDecayAgent(size_t num_actions, float beta) {
    set_beta(beta);
    set_n();
    num_available_actions_ = num_actions;
    q_ = std::vector<float>(num_actions, 0.);
    num_action_chosen_ = std::vector<size_t>(num_actions, 0);
    sum_reward_ = std::vector<float>(num_actions, 0.);
    sum_reward_square_ = std::vector<float>(num_actions, 0.);
    cumm_epsilon_action_prob_ = std::vector<float>(num_actions, 1.0 / num_actions);
    if (agent_info_file_) {
        fprintf(agent_info_file_, "action,reward,");
        for (size_t i = 0; i < num_available_actions_; ++i) {
            fprintf(agent_info_file_, "q%zu,", i);
        }
        for (size_t i = 0; i < num_available_actions_; ++i) {
            fprintf(agent_info_file_, "n%zu,", i);
        }
        fprintf(agent_info_file_, "\n");
    }
    set_epsilon_action_prob();
}

EpsilonDecayAgent::~EpsilonDecayAgent() {
    if (agent_info_file_) vtr::fclose(agent_info_file_);
}

void EpsilonDecayAgent::set_step(float gamma, int move_lim) {
    VTR_LOG("Setting decay step: %g\n", exp_alpha_);
    if (gamma < 0) {
        exp_alpha_ = -1; //Use sample average
    } else {
        //
        // For an exponentially wieghted average the fraction of total weight applied to
        // to moves which occured > K moves ago is:
        //
        //      gamma = (1 - alpha)^K
        //
        // If we treat K as the number of moves per temperature (move_lim) then gamma
        // is the fraction of weight applied to moves which occured > move_lim moves ago,
        // and given a target gamma we can explicitly calcualte the alpha step-size
        // required by the agent:
        //
        //     alpha = 1 - e^(log(gamma) / K)
        //
        float alpha = 1 - std::exp(std::log(gamma) / move_lim);
        exp_alpha_ = alpha;
    }
}

size_t EpsilonDecayAgent::propose_action() {
    size_t action = 0;
    if (vtr::frand() < 1 / (1 + n_ * beta_)) {
        /* Explore
         * With probability epsilon, choose randomly amongst all move types */
        float p = vtr::frand();
        auto itr = std::lower_bound(cumm_epsilon_action_prob_.begin(), cumm_epsilon_action_prob_.end(), p);
        action = itr - cumm_epsilon_action_prob_.begin();
    } else {
        /* Greedy (Exploit)
         * For probability 1-epsilon, choose the greedy action */
        auto itr = std::max_element(q_.begin(), q_.end());
        VTR_ASSERT(itr != q_.end());
        action = itr - q_.begin();
    }
    VTR_ASSERT(action < num_available_actions_);

    last_action_ = action;
    return action;
}


void EpsilonDecayAgent::set_beta(float beta) {
    VTR_LOG("Setting decay beta: %g\n", beta);
    beta_ = beta;
}

void EpsilonDecayAgent::set_n() {
    n_ = 0;
}

void EpsilonDecayAgent::set_epsilon_action_prob() {
    //initialize to equal probabilities
    std::vector<float> epsilon_prob(num_available_actions_, 1.0 / num_available_actions_);
    n_++;
    float accum = 0;
    for (size_t i = 0; i < num_available_actions_; ++i) {
        accum += epsilon_prob[i];
        cumm_epsilon_action_prob_[i] = accum;
    }
}

/*                                  *
 *                                  *
 *  UCB     agent implementation    *
 *                                  *
 *                                  */
UCBAgent::UCBAgent(size_t num_actions, float c) {
    set_c(c);
    num_available_actions_ = num_actions;
    q_ = std::vector<float>(num_actions, 0.);
    sum_reward_ = std::vector<float>(num_actions, 0.);
    sum_reward_square_ = std::vector<float>(num_actions, 0.);
    num_action_chosen_ = std::vector<size_t>(num_actions, 0);
    Decay_N_ = std::vector<float>(num_actions, 0.);
    t_ = 0;
    max_reward_ = 0;
    if (agent_info_file_) {
        fprintf(agent_info_file_, "action,reward,");
        for (size_t i = 0; i < num_available_actions_; ++i) {
            fprintf(agent_info_file_, "q%zu,", i);
        }
        for (size_t i = 0; i < num_available_actions_; ++i) {
            fprintf(agent_info_file_, "n%zu,", i);
        }
        fprintf(agent_info_file_, "\n");
        fprintf(agent_info_file_, "step");
    }
}

UCBAgent::~UCBAgent() {
    if (agent_info_file_) vtr::fclose(agent_info_file_);
}

void UCBAgent::process_outcome(double reward, e_reward_function reward_fun) {
    ++num_action_chosen_[last_action_];
    t_++;
    if (reward_fun == RUNTIME_AWARE || reward_fun == WL_BIASED_RUNTIME_AWARE)
        reward /= time_elapsed_[last_action_];
    //Determine step size
    float step = 0.;
    if (reward > max_reward_) {
        max_reward_ = reward;
    }
    if (exp_alpha_ < 0.) {
        step = 1. / num_action_chosen_[last_action_]; //Incremental average
    } else if (exp_alpha_ <= 1) {
        step = exp_alpha_; //Exponentially wieghted average
    } else {
        VTR_ASSERT_MSG(false, "Invalid step size");
    }

    //Based on the outcome how much should our estimate of q change?
    //float delta_q = step * (reward - q_[last_action_]);

    //Update the estimated value of the last action
    //q_[last_action_] += delta_q;

    // Update the sum of reward
    //sum_reward_[last_action_] = sum_reward_[last_action_] + reward;
    // Step mode
    for (size_t i = 0; i < num_available_actions_; i++) {
        sum_reward_[i] = sum_reward_[i] * step;
        Decay_N_[i] = Decay_N_[i] * step;
    }
    if (max_reward_ != 0) {
        reward = reward / max_reward_;
    }
    sum_reward_[last_action_] += reward;
    Decay_N_[last_action_] += 1;
    //sum_reward_square_[last_action_] = sum_reward_square_[last_action_] + std::pow(reward * step, 2);
    if (agent_info_file_) {
        fprintf(agent_info_file_, "%zu,", last_action_);
        fprintf(agent_info_file_, "%g,", reward);

        for (size_t i = 0; i < num_available_actions_; ++i) {
            fprintf(agent_info_file_, "%g,", q_[i]);
        }

        for (size_t i = 0; i < num_available_actions_; ++i) {
            fprintf(agent_info_file_, "%zu", num_action_chosen_[i]);
            if (i != num_available_actions_ - 1) {
                fprintf(agent_info_file_, ",");
            }
        }
        fprintf(agent_info_file_,",%g", step);
        fprintf(agent_info_file_, "\n");
    }
}

void UCBAgent::set_c(float c) {
    c_ = c;
}

void UCBAgent::set_step(float gamma, int move_lim) {
    VTR_LOG("Setting decay step: %g\n", exp_alpha_);
    if (gamma < 0) {
        exp_alpha_ = -1; //Use sample average
    } else {
        //
        // For an exponentially wieghted average the fraction of total weight applied to
        // to moves which occured > K moves ago is:
        //
        //      gamma = (1 - alpha)^K
        //
        // If we treat K as the number of moves per temperature (move_lim) then gamma
        // is the fraction of weight applied to moves which occured > move_lim moves ago,
        // and given a target gamma we can explicitly calcualte the alpha step-size
        // required by the agent:
        //
        //     alpha = 1 - e^(log(gamma) / K)
        //
        /*
        float alpha = 1 - std::exp(std::log(gamma) / move_lim);
        exp_alpha_ = alpha;
        */
        VTR_LOG("move_lim: %g\n", move_lim);
        exp_alpha_ = gamma;
    }
}

// choose an action
size_t UCBAgent::propose_action() {
    size_t action = 0;
    if (t_ < num_available_actions_) {
        action = t_;
    }
    else {
        // Find the max q value
        update_q();
        /*
        std::vector<float> p_ = std::vector<float>(num_available_actions_, 0.);;
        for (size_t i = 0; i < num_available_actions_; i++) {
            p_[i] = q_[i] + c_ * std::sqrt(std::log(t_ * std::pow(std::log(t_), 2) + 1) / num_action_chosen_[i]);
        }*/
        auto itr = std::max_element(q_.begin(), q_.end());
        VTR_ASSERT(itr != q_.end());
        action = itr - q_.begin();
    }
    VTR_ASSERT(action < num_available_actions_);

    last_action_ = action;
    return action;
}

//Update UCB values
void UCBAgent::update_q() {
    float decay_t_ = 0;
    for (size_t i = 0; i < num_available_actions_; i++) {
        decay_t_ += Decay_N_[i];
    }
    for (size_t i = 0; i < num_available_actions_; i++) {
        // f(t) = t
        //q_[i] = sum_reward_[i] / num_action_chosen_[i] + c_ * std::sqrt(std::log(t_) / num_action_chosen_[i]);
        // f(t) = 1 + t * log(t)^2
        //float mean = sum_reward_[i] / Decay_N_[i];
        //float V_ = std::max(mean*(1-mean), float(0.002));
        q_[i] = sum_reward_[i] / Decay_N_[i] + std::sqrt(c_ * std::log(decay_t_) /  Decay_N_[i]);
    }
}

/*                                  *
 *                                  *
 *  UCB     agent implementation    *
 *                                  *
 *                                  */
UCB1_Agent::UCB1_Agent(size_t num_actions, float c) {
    set_c(c);
    num_available_actions_ = num_actions;
    q_ = std::vector<float>(num_actions, 0.);
    sum_reward_ = std::vector<float>(num_actions, 0.);
    sum_reward_square_ = std::vector<float>(num_actions, 0.);
    num_action_chosen_ = std::vector<size_t>(num_actions, 0);
    SW_reward_ = std::vector<float>((int) c_, 0.);
    SW_action_ = std::vector<size_t>((int) c_, (size_t) 0);
    SW_num_action_chosen_ = std::vector<size_t>(num_actions, 0);
    SW_count_ = 0;
    t_ = 0;

    if (agent_info_file_) {
        fprintf(agent_info_file_, "action,reward,");
        for (size_t i = 0; i < num_available_actions_; ++i) {
            fprintf(agent_info_file_, "q%zu,", i);
        }
        for (size_t i = 0; i < num_available_actions_; ++i) {
            fprintf(agent_info_file_, "n%zu,", i);
        }
        fprintf(agent_info_file_, "\n");
    }
}

/*
Sliding-window UCB
*/
UCB1_Agent::~UCB1_Agent() {
    if (agent_info_file_) vtr::fclose(agent_info_file_);
}

void UCB1_Agent::process_outcome(double reward, e_reward_function reward_fun) {
    ++num_action_chosen_[last_action_];
    t_++;
    if (reward_fun == RUNTIME_AWARE || reward_fun == WL_BIASED_RUNTIME_AWARE)
        reward /= time_elapsed_[last_action_];
    //Determine step size
    float step = 0.;
    if (exp_alpha_ < 0.) {
        step = 1. / num_action_chosen_[last_action_]; //Incremental average
    } else if (exp_alpha_ >= 0) {
        step = exp_alpha_; //Exponentially wieghted average
    } else {
        VTR_ASSERT_MSG(false, "Invalid step size");
    }

    // Update the sum of reward
    //sum_reward_[last_action_] = sum_reward_[last_action_] + reward;
    // Step mode
    if (SW_count_ < c_) {
        // update X_t and N_t
        sum_reward_[last_action_] += reward;
        SW_num_action_chosen_[last_action_]++;

        // add to the sliding window
        SW_reward_[SW_count_] = reward;
        SW_action_[SW_count_] = last_action_;
        SW_count_++;
    }
    else {
        SW_num_action_chosen_[SW_action_[0]]--; // Decrease the expelled element's action number by one
        sum_reward_[SW_action_[0]] -= SW_reward_[0]; // Decrease the sum of reward by the first element of SW_reward

        // update sliding window
        //std::shift_left(begin(SW_reward_), end(SW_reward_), 1);
        SW_reward_.erase(SW_reward_.begin());
        //std::shift_left(begin(SW_action_), end(SW_action_), 1);
        SW_action_.erase(SW_action_.begin());
        //SW_reward_[c_ - 1] = reward;
        SW_reward_.push_back(reward);
        //SW_action_[c_ - 1] = last_action_;
        SW_action_.push_back(last_action_);

        // update X_t and N_t
        sum_reward_[last_action_] += reward;
        SW_num_action_chosen_[last_action_];
    }
    /* UCB_TUNED1
    sum_reward_[last_action_] = sum_reward_[last_action_] + reward * step;
    sum_reward_square_[last_action_] = sum_reward_square_[last_action_] + std::pow(reward * step, 2);
    */
    if (agent_info_file_) {
        fprintf(agent_info_file_, "%zu,", last_action_);
        fprintf(agent_info_file_, "%g,", reward);

        for (size_t i = 0; i < num_available_actions_; ++i) {
            fprintf(agent_info_file_, "%g,", q_[i]);
        }

        for (size_t i = 0; i < num_available_actions_; ++i) {
            fprintf(agent_info_file_, "%zu", num_action_chosen_[i]);
            if (i != num_available_actions_ - 1) {
                fprintf(agent_info_file_, ",");
            }
        }
        fprintf(agent_info_file_, "\n");
    }
}

void UCB1_Agent::set_c(float c) {
    c_ = c;
}

void UCB1_Agent::set_step(float gamma, int move_lim) {
    VTR_LOG("Setting decay step: %g\n", exp_alpha_);
    if (gamma < 0) {
        exp_alpha_ = -1; //Use sample average
    } else {
        //
        // For an exponentially wieghted average the fraction of total weight applied to
        // to moves which occured > K moves ago is:
        //
        //      gamma = (1 - alpha)^K
        //
        // If we treat K as the number of moves per temperature (move_lim) then gamma
        // is the fraction of weight applied to moves which occured > move_lim moves ago,
        // and given a target gamma we can explicitly calcualte the alpha step-size
        // required by the agent:
        //
        //     alpha = 1 - e^(log(gamma) / K)
        //
        exp_alpha_ = gamma;
    }
}

// choose an action
size_t UCB1_Agent::propose_action() {
    size_t action = 0;
    if (t_ < num_available_actions_) {
        action = t_;
    }
    else {
        // Find the max q value
        update_q();
        auto itr = std::max_element(q_.begin(), q_.end());
        VTR_ASSERT(itr != q_.end());
        action = itr - q_.begin();
    }
    VTR_ASSERT(action < num_available_actions_);

    last_action_ = action;
    return action;
}

//Update UCB values
void UCB1_Agent::update_q() {
    for (size_t i = 0; i < num_available_actions_; i++) {
        // f(t) = t
        //q_[i] = sum_reward_[i] / num_action_chosen_[i] + c_ * std::sqrt(std::log(t_) / num_action_chosen_[i]);
        // f(t) = 1 + t * log(t)^2
        /* UCB1-TUNED
        float V = sum_reward_square_[i] / num_action_chosen_[i] - std::pow(sum_reward_[i] / num_action_chosen_[i], 2) + std::sqrt(2 * std::log(t_) / num_action_chosen_[i]);
        V = std::min(V, float(0.25));
        q_[i] = sum_reward_[i] / num_action_chosen_[i] + c_ * std::sqrt(std::log(t_) / num_action_chosen_[i] * V);
        */
        /*
        SW-UCB
        */
        if (SW_num_action_chosen_[i] == 0) {
            q_[i] = std::numeric_limits<float>::max();
        }
        else {
            q_[i] = sum_reward_[i] / SW_num_action_chosen_[i] + std::sqrt(std::log(std::min((int) t_, (int) SW_count_)) * exp_alpha_ / SW_num_action_chosen_[i]);
        }
    }
}


/*                                  *
 *                                  *
 *  EXP3Agent agent implementation    *
 *                                  *
 *                                  */
void EXP3Agent::process_outcome(double reward, e_reward_function reward_fun) {
    ++num_action_chosen_[last_action_];
    t_++;
    if (reward_fun == RUNTIME_AWARE || reward_fun == WL_BIASED_RUNTIME_AWARE)
        reward /= time_elapsed_[last_action_];
    //Determine step size
    float step = 0.;
    if (exp_alpha_ < 0.) {
        step = 1. / num_action_chosen_[last_action_]; //Incremental average
    } else if (exp_alpha_ <= 1) {
        step = exp_alpha_; //Exponentially wieghted average
    } else {
        VTR_ASSERT_MSG(false, "Invalid step size");
    }
    reward = reward * 10000 * step;
    if (reward == 0.) {
        reward = -0.000000001 * 10000;
    }
    w_[last_action_] = w_[last_action_] * std::exp(gamma_ * reward / action_prob_[last_action_] / num_available_actions_); // EXP3
    /*
    //Based on the outcome how much should our estimate of q change?
    float delta_q = step * (reward - q_[last_action_]);

    //Update the estimated value of the last action
    q_[last_action_] += delta_q;

    // Update the sum of reward
    //sum_reward_[last_action_] = sum_reward_[last_action_] + reward;
    // Step mode
    sum_reward_[last_action_] = sum_reward_[last_action_] + reward * step;
    sum_reward_square_[last_action_] = sum_reward_square_[last_action_] + std::pow(reward * step, 2);
    */
    if (agent_info_file_) {
        fprintf(agent_info_file_, "%zu,", last_action_);
        fprintf(agent_info_file_, "%g,", reward);

        for (size_t i = 0; i < num_available_actions_; ++i) {
            fprintf(agent_info_file_, "%g,", q_[i]);
        }

        for (size_t i = 0; i < num_available_actions_; ++i) {
            fprintf(agent_info_file_, "%zu", num_action_chosen_[i]);
            if (i != num_available_actions_ - 1) {
                fprintf(agent_info_file_, ",");
            }
        }
        fprintf(agent_info_file_, "\n");
    }
}

EXP3Agent::EXP3Agent(size_t num_actions, float gamma) {
    num_available_actions_ = num_actions;
    q_ = std::vector<float>(num_actions, 0.);
    exp_q_ = std::vector<float>(num_actions, 0.);
    num_action_chosen_ = std::vector<size_t>(num_actions, 0);
    action_prob_ = std::vector<float>(num_actions, 0.);
    sum_reward_ = std::vector<float>(num_actions, 0.);
    sum_reward_square_ = std::vector<float>(num_actions, 0.);
    w_ = std::vector<float>(num_actions, 1.);
    gamma_ = gamma;
    cumm_action_prob_ = std::vector<float>(num_actions);
    if (agent_info_file_) {
        fprintf(agent_info_file_, "action,reward,");
        for (size_t i = 0; i < num_available_actions_; ++i) {
            fprintf(agent_info_file_, "q%zu,", i);
        }
        for (size_t i = 0; i < num_available_actions_; ++i) {
            fprintf(agent_info_file_, "n%zu,", i);
        }
        fprintf(agent_info_file_, "\n");
    }
    set_action_prob();
    //agent_info_file_ = vtr::fopen("agent_info.txt", "w");
}

EXP3Agent::~EXP3Agent() {
    if (agent_info_file_) vtr::fclose(agent_info_file_);
}

size_t EXP3Agent::propose_action() {
    set_action_prob();
    size_t action = 0;
    float p = vtr::frand();
    auto itr = std::lower_bound(cumm_action_prob_.begin(), cumm_action_prob_.end(), p);
    action = itr - cumm_action_prob_.begin();
    //To take care that the last element in cumm_action_prob_ might be less than 1 by a small value
    if (action == num_available_actions_)
        action = num_available_actions_ - 1;
    VTR_ASSERT(action < num_available_actions_);

    last_action_ = action;
    return action;
}

void EXP3Agent::set_action_prob() {
    float w_sum_ = 0.;
    for (size_t i = 0; i < num_available_actions_; ++i) {
        w_sum_ += w_[i];
    }
    std::vector<float> p_ = std::vector<float>(num_available_actions_, 0.);
    for (size_t i = 0; i < num_available_actions_; ++i) {
        action_prob_[i] = (1 - gamma_) * w_[i] / w_sum_ + gamma_ / num_available_actions_;
    }

    //calulcate the accumulative action probability of each action
    // e.g. if we have 5 actions with equal probability of 0.2, the cumm_action_prob will be {0.2,0.4,0.6,0.8,1.0}
    float accum = 0;
    for (size_t i = 0; i < num_available_actions_; ++i) {
        accum += action_prob_[i];
        cumm_action_prob_[i] = accum;
    }
}

void EXP3Agent::set_step(float gamma, int move_lim) {
    if (gamma < 0) {
        exp_alpha_ = -1; //Use sample average
    } else {
        //
        // For an exponentially weighted average the fraction of total weight applied
        // to moves which occured > K moves ago is:
        //
        //      gamma = (1 - alpha)^K
        //
        // If we treat K as the number of moves per temperature (move_lim) then gamma
        // is the fraction of weight applied to moves which occured > move_lim moves ago,
        // and given a target gamma we can explicitly calcualte the alpha step-size
        // required by the agent:
        //
        //     alpha = 1 - e^(log(gamma) / K)
        //
        float alpha = 1 - std::exp(std::log(gamma) / move_lim);
        exp_alpha_ = alpha;
    }
}