#ifndef VPR_SIMPLERL_MOVE_GEN_H
#define VPR_SIMPLERL_MOVE_GEN_H
#include "move_generator.h"
#include "median_move_generator.h"
#include "weighted_median_move_generator.h"
#include "feasible_region_move_generator.h"
#include "weighted_centroid_move_generator.h"
#include "uniform_move_generator.h"
#include "critical_uniform_move_generator.h"
#include "centroid_move_generator.h"
#include <stdio.h>
#include <zmq.hpp>

/**
 * @brief KArmedBanditAgent is the base class for RL agents that target the k-armed bandit problems
 */
class KArmedBanditAgent {
  public:
    virtual ~KArmedBanditAgent() {}
    virtual size_t propose_action() = 0;
    virtual void process_outcome(double /*reward*/, e_reward_function /*reward_fun*/) {
    }

  protected:
    float exp_alpha_ = -1;                  //Step size for q_ updates (< 0 implies use incremental average)
    size_t num_available_actions_;          //Number of arms of the karmed bandit problem (k)
    std::vector<size_t> num_action_chosen_; //Number of times each arm has been pulled (n)
    std::vector<float> q_;                  //Estimated value of each arm (Q)
    size_t last_action_ = 0;                //type of the last action (move type) proposed
    /* Ratios of the average runtime to calculate each move type              */
    /* These ratios are useful for different reward functions                 *
     * The vector is calculated by averaging many runs on different circuits  */
    std::vector<double> time_elapsed_{1.0, 3.6, 5.4, 2.5, 2.1, 0.8, 2.2};
    size_t t_ = 0;
    std::vector<float> sum_reward_;       //The sum of reward
    std::vector<float> sum_reward_square_;       //The sum of reward's square
    //FILE* agent_info_file_ = fopen("agent.txt", "w");
    FILE* agent_info_file_ = NULL;
    
};

/**
 * @brief Epsilon-greedy agent implementation to address K-armed bandit problems
 *
 * In this class, the RL agent addresses the exploration-exploitation dilemma using epsilon- greedy
 * where the agent chooses the best action (the greedy) most of the time. For a small fraction of the time 
 * (epsilon), it randomly chooses any action to explore the solution space.
 */
class EpsilonGreedyAgent : public KArmedBanditAgent {
  public:
    EpsilonGreedyAgent(size_t num_actions, float epsilon);
    ~EpsilonGreedyAgent();
    void process_outcome(double reward, e_reward_function reward_fun) override; //Updates the agent based on the reward of the last proposed action
    size_t propose_action() override; //Returns the type of the next action the agent wishes to perform

  public:
    void set_epsilon(float epsilon);
    void set_epsilon_action_prob();
    void set_step(float gamma, int move_lim);

  private:
    float epsilon_ = 0.1;                         //How often to perform a non-greedy exploration action
    std::vector<float> cumm_epsilon_action_prob_; //The accumulative probability of choosing each action
};

/**
 * @brief Epsilon-decay agent implementation to address K-armed bandit problems
 *
 * In this class, the RL agent addresses the exploration-exploitation dilemma using epsilon-decay
 * where the agent chooses the best action (the greedy) most of the time. For a small decreasing fraction of the time 
 * (epsilon), it randomly chooses any action to explore the solution space.
 */

class EpsilonDecayAgent : public KArmedBanditAgent {
  public:
    EpsilonDecayAgent(size_t num_actions, float beta);
    ~EpsilonDecayAgent();
    void process_outcome(double reward, e_reward_function reward_fun) override; //Updates the agent based on the reward of the last proposed action
    size_t propose_action() override; //Returns the type of the next action the agent wishes to perform

  public:
    void set_beta(float beta);
    void set_n();
    void set_epsilon_action_prob();
    void set_step(float gamma, int move_lim);

  private:
    float beta_ = 0.01;
    int n_ = 0;
    std::vector<float> cumm_epsilon_action_prob_; //The accumulative probability of choosing each action
};

/**
 * @brief Softmax agent implementation to address K-armed bandit problems
 *
 * In this class, the RL agent adresses the exploration-exploitation dilemma using Softmax
 * where the probability of choosing each action propotional to the estimated Q value of
 * this action. 
 */
class SoftmaxAgent : public KArmedBanditAgent {
  public:
    SoftmaxAgent(size_t num_actions);
    ~SoftmaxAgent();

    void process_outcome(double reward, e_reward_function reward_fun) override; //Updates the agent based on the reward of the last proposed action
    size_t propose_action() override; //Returns the type of the next action the agent wishes to perform

  public:
    void set_action_prob();
    void set_step(float gamma, int move_lim);

  private:
    std::vector<float> exp_q_;            //The clipped and scaled exponential of the estimated Q value for each action
    std::vector<float> action_prob_;      //The probability of choosing each action
    std::vector<float> cumm_action_prob_; //The accumulative probability of choosing each action
};

/**
 * @brief UCB agent implementation to address K-armed bandit problems
 *
 * In this class, the RL agent adresses the exploration-exploitation dilemma using UCB
 * 
 * 
 */
class UCBAgent : public KArmedBanditAgent {
  public:
    UCBAgent(size_t num_actions, float c);
    ~UCBAgent();

    void process_outcome(double reward, e_reward_function reward_fun) override; //Updates the agent based on the reward of the last proposed action
    size_t propose_action() override; //Returns the type of the next action the agent wishes to perform
  public:
    void set_step(float gamma, int move_lim);
    void set_c(float c);
    void update_q();

  private:
    float c_ = 0.01;
    std::vector<float> Decay_N_;
    float max_reward_;
};

/**
 * @brief UCB agent implementation to address K-armed bandit problems
 *
 * In this class, the RL agent adresses the exploration-exploitation dilemma using UCB
 * 
 * 
 */
class UCB1_Agent : public KArmedBanditAgent {
  public:
    UCB1_Agent(size_t num_actions, float c);
    ~UCB1_Agent();

    void process_outcome(double reward, e_reward_function reward_fun) override; //Updates the agent based on the reward of the last proposed action
    size_t propose_action() override; //Returns the type of the next action the agent wishes to perform
  public:
    void set_step(float gamma, int move_lim);
    void set_c(float c);
    void update_q();

  private:
    float c_ = 0.01;
    std::vector<float> SW_reward_;
    std::vector<size_t> SW_action_;
    std::vector<size_t> SW_num_action_chosen_;
    int SW_count_;
    float max_reward_;
};

class EXP3Agent : public KArmedBanditAgent {
  public:
    EXP3Agent(size_t num_actions, float gamma);
    ~EXP3Agent();

    void process_outcome(double reward, e_reward_function reward_fun) override; //Updates the agent based on the reward of the last proposed action
    size_t propose_action() override; //Returns the type of the next action the agent wishes to perform
  public:
    void set_action_prob();
    void set_step(float gamma, int move_lim);

  private:
    std::vector<float> w_;
    float gamma_;
    std::vector<float> exp_q_;            //The clipped and scaled exponential of the estimated Q value for each action
    std::vector<float> action_prob_;      //The probability of choosing each action
    std::vector<float> cumm_action_prob_; //The accumulative probability of choosing each action
};

class UCBCAgent : public KArmedBanditAgent {
  public:
    UCBCAgent(size_t num_actions, float c);
    ~UCBCAgent();

    void process_outcome(double reward, e_reward_function reward_fun) override; //Updates the agent based on the reward of the last proposed action
    size_t propose_action() override; //Returns the type of the next action the agent wishes to perform
  public:
    void set_step(float gamma, int move_lim);
    void update_q();

  private:
    float c_ = 0.01;
    std::vector<float> decay_N_;
    float max_reward_;
    size_t num_availabel_clusters_;
    std::vector<float> q_cluster_;
    std::vector<std::vector<float>> q_arm_;
    std::vector<std::vector<int>> cluster_information_;
};

class MOSSAgent : public KArmedBanditAgent {
  public:
    MOSSAgent(size_t num_actions, float c);
    ~MOSSAgent();

    void process_outcome(double reward, e_reward_function reward_fun) override; //Updates the agent based on the reward of the last proposed action
    size_t propose_action() override; //Returns the type of the next action the agent wishes to perform
  public:
    void set_step(float gamma, int move_lim);
    void set_c(float c);
    void update_q();

  private:
    float c_ = 0.01;
    std::vector<float> Decay_N_;
    float max_reward_;
};

/**
 * @brief This class represents the move generator that uses simple RL agent
 * 
 * It is a derived class from MoveGenerator that utilizes a simple RL agent to
 * dynamically select the optimal probability of each action (move type)
 * 
 */
class SimpleRLMoveGenerator : public MoveGenerator {
  private:
    std::vector<std::unique_ptr<MoveGenerator>> avail_moves; // list of pointers to the available move generators (the different move types)
    std::unique_ptr<KArmedBanditAgent> karmed_bandit_agent;  // a pointer to the specific agent used (e.g. Softmax)

  public:
    // constructors using a pointer to the agent used
    SimpleRLMoveGenerator(std::unique_ptr<EpsilonGreedyAgent>& agent);
    SimpleRLMoveGenerator(std::unique_ptr<SoftmaxAgent>& agent);
    SimpleRLMoveGenerator(std::unique_ptr<EpsilonDecayAgent>& agent);
    SimpleRLMoveGenerator(std::unique_ptr<UCBAgent>& agent);
    SimpleRLMoveGenerator(std::unique_ptr<UCB1_Agent>& agent);
    SimpleRLMoveGenerator(std::unique_ptr<EXP3Agent>& agent);
    SimpleRLMoveGenerator(std::unique_ptr<UCBCAgent>& agent);
    SimpleRLMoveGenerator(std::unique_ptr<MOSSAgent>& agent);


    // Updates affected_blocks with the proposed move, while respecting the current rlim
    e_create_move propose_move(t_pl_blocks_to_be_moved& blocks_affected, e_move_type& move_type, float rlim, const t_placer_opts& placer_opts, const PlacerCriticalities* criticalities);
    e_create_move propose_move_with_type(t_pl_blocks_to_be_moved& /*blocks_affected*/, e_move_type& /*move_type*/, float /*rlim*/, const t_placer_opts& /*placer_opts*/, const PlacerCriticalities* /*criticalities*/, const char* /*blk_type_name*/) {return e_create_move::ABORT;}
    // Recieves feedback about the outcome of the previously proposed move
    void process_outcome(double reward, e_reward_function reward_fun, double /*delta_c*/, double /*delta_bb_cost_norm*/, double /*delta_timing_cost_norm*/);

};

/**
 * @brief This class represents the move generator that communicate with RL gym
 * 
 * It is a derived class from MoveGenerator that utilizes a RL gym to
 * dynamically select the optimal probability of each action (move type)
 * 
 */
class RLGymGenerator: public MoveGenerator {
  private:
    std::vector<std::unique_ptr<MoveGenerator>> avail_moves; // list of pointers to the available move generators (the different move types)
    zmq::context_t ctx;
    zmq::socket_t socket;
    size_t last_action_ = 0;
    std::vector<double> time_elapsed_{1.0, 3.6, 5.4, 2.5, 2.1, 0.8, 2.2};
    //std::vector<double> time_elapsed_{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    size_t num_available_actions_;
    std::set<std::string> blk_type_set;
    std::vector<std::string> blk_types;
    std::vector<int> blk_type_idx;
    std::map<std::string, int> blk_type_num;
    float elapsed_time = 0;
    bool reset_happened = false;
  public:
    RLGymGenerator(size_t num_actions, const t_placer_opts& placer_opts, int move_lim);
    ~RLGymGenerator();
    e_create_move propose_move(t_pl_blocks_to_be_moved& blocks_affected, e_move_type& move_type, float rlim, const t_placer_opts& placer_opts, const PlacerCriticalities* criticalities);
    void process_outcome(double reward, e_reward_function reward_fun, double delta_c, double delta_bb_cost_norm, double delta_timing_cost_norm);
    e_create_move propose_move_with_type(t_pl_blocks_to_be_moved& /*blocks_affected*/, e_move_type& /*move_type*/, float /*rlim*/, const t_placer_opts& /*placer_opts*/, const PlacerCriticalities* /*criticalities*/, const char* /*blk_type_name*/) {return e_create_move::ABORT;}
    void find_all_types();
    void reset_agent();
    void stage2();
};


#endif
