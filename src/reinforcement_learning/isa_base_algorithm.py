from abc import abstractmethod
import numpy as np
import os
import pickle
import torch

from torch.autograd import Variable
from gym_subgoal_automata.utils.subgoal_automaton import SubgoalAutomaton
from labelling_function.state_to_event_net import State2EventNet
from reinforcement_learning.learning_algorithm import LearningAlgorithm
from utils import utils
from ilasp.generator.ilasp_task_generator import generate_ilasp_task
from ilasp.parser import ilasp_solution_parser
from ilasp.solver.ilasp_solver import solve_ilasp_task


class ISAAlgorithmBase(LearningAlgorithm):
    """
    Generic class for the algorithms performing interleaving between RL and automata learning.
    """
    INITIAL_STATE_NAME = "u0"
    ACCEPTING_STATE_NAME = "u_acc"
    REJECTING_STATE_NAME = "u_rej"

    # whether to use the single state automaton (basic), load an existing solution (load) or use the target one (target)
    INITIAL_AUTOMATON_MODE = "initial_automaton"

    INTERLEAVED_FIELD = "interleaved_automaton_learning"           # whether RL is interleaved with ILASP automaton learner
    ILASP_TIMEOUT_FIELD = "ilasp_timeout"                          # time that ILASP has for finding a single automaton solution
    ILASP_VERSION_FIELD = "ilasp_version"                          # ILASP version to run
    ILASP_COMPUTE_MINIMAL = "ilasp_compute_minimal"                # whether to compute a minimal solution (an optimal is computed otherwise)
    STARTING_NUM_STATES_FIELD = "starting_num_states"              # number of states that the starting automaton has
    USE_RESTRICTED_OBSERVABLES = "use_restricted_observables"      # use the restricted set of observables (the ones that define the goal for the task)
    MAX_DISJUNCTION_SIZE = "max_disjunction_size"                  # maximum number of edges from one state to another
    MAX_BODY_LITERALS = "max_body_literals"                        # maximum number of literals that a learnt rule can have
    LEARN_ACYCLIC_GRAPH = "learn_acyclic_graph"                    # whether the target automata has cycles or not
    SYMMETRY_BREAKING_METHOD = "symmetry_breaking_method"          # which symmetry breaking method is used to break symmetries in the graph
    AVOID_LEARNING_ONLY_NEGATIVE = "avoid_learning_only_negative"  # whether to avoid learning labels made only of negative literals (e.g., ~n)
    PRIORITIZE_OPTIMAL_SOLUTIONS = "prioritize_optimal_solutions"  # prioritize some optimal solutions above others based on some weak constraints
    WAIT_FOR_GOAL_EXAMPLE = "wait_for_goal_example"                # whether an automaton is not learnt until a goal example is received

    USE_MAX_EPISODE_LENGTH_ANNEALING = "use_max_episode_length_annealing"  # whether to increase the maximum episode length as learning progresses
    INITIAL_MAX_EPISODE_LENGTH = "initial_max_episode_length"
    FINAL_MAX_EPISODE_LENGTH = "final_max_episode_length"

    USE_EXPERIENCE_REPLAY = "use_experience_replay"                  # whether to use the experience replay buffer for learning (automatically active for deep learning approach)
    EXPERIENCE_REPLAY_BUFFER_SIZE = "experience_replay_buffer_size"  # size of the ER buffer
    EXPERIENCE_REPLAY_BATCH_SIZE = "experience_replay_batch_size"    # size of the batches sampled from the ER buffer
    EXPERIENCE_REPLAY_START_SIZE = "experience_replay_start_size"    # size of the ER after which learning starts

    USE_DOUBLE_DQN = "use_double_dqn"                            # whether double DQN is used instead of simple DQN
    TARGET_NET_UPDATE_FREQUENCY = "target_net_update_frequency"  # how many steps happen between target DQN updates
    NUM_HIDDEN_LAYERS = "num_hidden_layers"                      # number of hidden layers that the network has
    NUM_NEURONS_PER_LAYER = "num_neurons_per_layer"              # number of neurons per hidden layer

    AUTOMATON_TASK_FOLDER = "automaton_tasks"  # folder where the automaton learning tasks are saved
    AUTOMATON_TASK_FILENAME = "task-%d.las"    # filename pattern of the automaton learning tasks

    AUTOMATON_SOLUTION_FOLDER = "automaton_solutions"  # folder where the solutions to automaton learning tasks are saved
    AUTOMATON_SOLUTION_FILENAME = "solution-%d.txt"    # filename pattern of the solutions to automaton learning tasks

    AUTOMATON_PLOT_FOLDER = "automaton_plots"          # folder where the graphical solutions to automaton learning tasks are saved
    AUTOMATON_PLOT_FILENAME = "plot-%d.png"            # filename pattern of the graphical solutions to automaton learning tasks

    AUTOMATON_LEARNING_EPISODES_FILENAME = "automaton_learning_episodes.txt"  # filename of the file containing the episodes where an automaton has been learned

    DYNAMIC_ENV_INPUT_VEC_SIZE = 52
    DYNAMIC_ENV_OUTPUT_VEC_SIZE = 21
    FROZEN_ENV_INPUT_VEC_SIZE = 28
    FROZEN_ENV_OUTPUT_VEC_SIZE = 6

    def __init__(self, tasks, num_tasks, export_folder_names, params, target_automata, binary_folder_name):
        super().__init__(tasks, num_tasks, export_folder_names, params)
        self.binary_folder_name = binary_folder_name

        self.initial_automaton_mode = utils.get_param(params, ISAAlgorithmBase.INITIAL_AUTOMATON_MODE, "basic")

        # interleaved automaton learning params
        self.interleaved_automaton_learning = utils.get_param(params, ISAAlgorithmBase.INTERLEAVED_FIELD, False)
        self.ilasp_timeout = utils.get_param(params, ISAAlgorithmBase.ILASP_TIMEOUT_FIELD, 120)
        self.ilasp_version = utils.get_param(params, ISAAlgorithmBase.ILASP_VERSION_FIELD, "2")
        self.ilasp_compute_minimal = utils.get_param(params, ISAAlgorithmBase.ILASP_COMPUTE_MINIMAL, False)
        self.num_starting_states = utils.get_param(params, ISAAlgorithmBase.STARTING_NUM_STATES_FIELD, 3)
        self.num_automaton_states = self.num_starting_states * np.ones(self.num_domains, dtype=np.int)
        self.use_restricted_observables = utils.get_param(params, ISAAlgorithmBase.USE_RESTRICTED_OBSERVABLES, False)
        self.max_disjunction_size = utils.get_param(params, ISAAlgorithmBase.MAX_DISJUNCTION_SIZE, 1)
        self.max_body_literals = utils.get_param(params, ISAAlgorithmBase.MAX_BODY_LITERALS, 1)
        self.learn_acyclic_graph = utils.get_param(params, ISAAlgorithmBase.LEARN_ACYCLIC_GRAPH, False)
        self.symmetry_breaking_method = utils.get_param(params, ISAAlgorithmBase.SYMMETRY_BREAKING_METHOD, None)
        self.avoid_learning_only_negative = utils.get_param(params, ISAAlgorithmBase.AVOID_LEARNING_ONLY_NEGATIVE, False)
        self.prioritize_optimal_solutions = utils.get_param(params, ISAAlgorithmBase.PRIORITIZE_OPTIMAL_SOLUTIONS, False)
        self.wait_for_goal_example = utils.get_param(params, ISAAlgorithmBase.WAIT_FOR_GOAL_EXAMPLE, True)
        self.has_observed_goal_example = np.zeros(self.num_domains, dtype=np.bool)

        # maximum episode annealing parameters
        self.use_max_episode_length_annealing = utils.get_param(params, ISAAlgorithmBase.USE_MAX_EPISODE_LENGTH_ANNEALING, False)
        self.final_max_episode_length = utils.get_param(params, ISAAlgorithmBase.FINAL_MAX_EPISODE_LENGTH, 100)
        if self.use_max_episode_length_annealing:
            self.initial_max_episode_length = utils.get_param(params, ISAAlgorithmBase.INITIAL_MAX_EPISODE_LENGTH, 100)
            self.max_episode_length = self.initial_max_episode_length
            self.max_episode_length_increase_rate = (self.final_max_episode_length - self.max_episode_length) / self.num_episodes

        # experience replay
        self.use_experience_replay = utils.get_param(params, ISAAlgorithmBase.USE_EXPERIENCE_REPLAY, False) or not self.is_tabular_case
        self.experience_replay_buffer_size = utils.get_param(params, ISAAlgorithmBase.EXPERIENCE_REPLAY_BUFFER_SIZE, 50000)
        self.experience_replay_batch_size = utils.get_param(params, ISAAlgorithmBase.EXPERIENCE_REPLAY_BATCH_SIZE, 32)
        self.experience_replay_start_size = utils.get_param(params, ISAAlgorithmBase.EXPERIENCE_REPLAY_START_SIZE, 1000)

        # deep q-learning
        self.use_double_dqn = utils.get_param(params, ISAAlgorithmBase.USE_DOUBLE_DQN, True)
        self.num_layers = utils.get_param(params, ISAAlgorithmBase.NUM_HIDDEN_LAYERS, 6)
        self.num_neurons_per_layer = utils.get_param(params, ISAAlgorithmBase.NUM_NEURONS_PER_LAYER, 64)
        self.target_net_update_frequency = utils.get_param(params, ISAAlgorithmBase.TARGET_NET_UPDATE_FREQUENCY, 100)

        # set of automata per domain
        self.automata = None
        self._set_automata(target_automata)

        # sets of examples (goal, deadend and incomplete)
        self.goal_examples = None
        self.dend_examples = None
        self.inc_examples = None
        self._reset_examples()

        # keep track of the number of learnt automata per domain
        self.automaton_counters = np.zeros(self.num_domains, dtype=np.int)
        self.automaton_learning_episodes = [[] for _ in range(self.num_domains)]

        if self.train_model:  # if the tasks are learnt, remove previous folders if they exist
            utils.rm_dirs(self.get_automaton_task_folders())
            utils.rm_dirs(self.get_automaton_solution_folders())
            utils.rm_dirs(self.get_automaton_plot_folders())
            utils.rm_files(self.get_automaton_learning_episodes_files())

    '''
    Learning Loop (main loop, what happens when an episode ends, changes or was not completed)
    '''
    def run(self, loaded_checkpoint=False):
        super().run(loaded_checkpoint)
        if self.interleaved_automaton_learning:
            self._write_automaton_learning_episodes()

    def _run_episode(self, domain_id, task_id):
        task = self._get_task(domain_id, task_id)  # get the task to learn

        # Initalise labelling function model and other valuable metadata about said model
        labelling_function, model_metrics, events_captured = self._initialise_labelling_function_and_metadata(task)

        # initialize reward and steps counters, histories and reset the task to its initial state
        total_reward, episode_length = 0, 0
        observation_history, compressed_observation_history = [], []
        current_state = task.reset()

        # get initial observations from the labelling function model and initialise histories
        # initial_observations = self._get_task_observations_from_env(task)
        initial_observations = self._get_task_observations_from_model(task, labelling_function, model_metrics, events_captured, current_state)
        self._update_histories(observation_history, compressed_observation_history, initial_observations)

        # get actual initial automaton state (performs verification that there is only one possible initial state!)
        current_automaton_state = self._get_initial_automaton_state_successors(domain_id, initial_observations)

        # update the automaton if the initial state achieves the goal and the example is not covered
        if self.interleaved_automaton_learning and self._can_learn_new_automaton(domain_id, task):
            updated_automaton = self._perform_interleaved_automaton_learning(task, domain_id,
                                                                             current_automaton_state,
                                                                             observation_history,
                                                                             compressed_observation_history, events_captured)
            if updated_automaton:  # get the actual initial state as done before
                current_automaton_state = self._get_initial_automaton_state_successors(domain_id, initial_observations)

        # whether the episode execution must be stopped (an automaton is learnt in the middle)
        interrupt_episode = False
        automaton = self.automata[domain_id]

        while not task.is_terminal() and episode_length < self.max_episode_length and not interrupt_episode:
            current_automaton_state_id = automaton.get_state_id(current_automaton_state)
            action = self._choose_action(domain_id, task_id, current_state, automaton, current_automaton_state_id)
            next_state, reward, is_terminal, _ = task.step(action)
            # observations = self._get_task_observations_from_env(task)
            observations = self._get_task_observations_from_model(task, labelling_function, model_metrics, events_captured, next_state)

            # whether observations have changed or not is important for QRM when using compressed traces
            observations_changed = self._update_histories(observation_history, compressed_observation_history, observations)

            if self.train_model:
                self._update_q_functions(task_id, current_state, action, next_state, is_terminal, observations, observations_changed)

            next_automaton_state = self._get_next_automaton_state(self.automata[domain_id], current_automaton_state,
                                                                  observations, observations_changed)

            # episode has to be interrupted if an automaton is learnt
            if not interrupt_episode and self.interleaved_automaton_learning and self._can_learn_new_automaton(domain_id, task):
                interrupt_episode = self._perform_interleaved_automaton_learning(task, domain_id, next_automaton_state,
                                                                                 observation_history,
                                                                                 compressed_observation_history, events_captured)

            if not interrupt_episode:
                automaton = self.automata[domain_id]
                total_reward += reward

                self._on_performed_step(domain_id, task_id, next_state, reward, is_terminal, observations, automaton,
                                        current_automaton_state, next_automaton_state, episode_length)

            # update current environment and automaton states and increase episode length
            current_state = next_state
            current_automaton_state = next_automaton_state
            episode_length += 1

        completed_episode = not interrupt_episode

        return completed_episode, total_reward, episode_length, task.is_terminal(), observation_history, compressed_observation_history

    def _initialise_labelling_function_and_metadata(self, task):
        model_sub_dir = ""
        input_vec_size = 0
        output_vec_size = 0
        if task.use_velocities:
            model_sub_dir = "dynamic/"
            input_vec_size = ISAAlgorithmBase.DYNAMIC_ENV_INPUT_VEC_SIZE
            output_vec_size = ISAAlgorithmBase.DYNAMIC_ENV_OUTPUT_VEC_SIZE
        else:
            model_sub_dir = "frozen/"
            input_vec_size = ISAAlgorithmBase.FROZEN_ENV_INPUT_VEC_SIZE
            output_vec_size = ISAAlgorithmBase.FROZEN_ENV_OUTPUT_VEC_SIZE
        num_layers = 6
        num_neurons = 64
        labelling_function = State2EventNet(input_vec_size, output_vec_size, num_layers, num_neurons)

        main_dir = "labelling_function/"
        model_dir = main_dir + model_sub_dir
        model_loc = model_dir + "final_model.pth"
        model_metrics_loc = model_dir + "final_model_metrics.pkl"
        events_captured_loc = model_dir + "events_captured.pkl"

        labelling_function.load_state_dict(torch.load(model_loc, map_location=torch.device('cpu')))
        with open(model_metrics_loc, "rb") as f:
            model_metrics = pickle.load(f)
        with open(events_captured_loc, "rb") as g:
            events_captured = pickle.load(g)
        events_captured_filtered = sorted(list(filter(lambda pair: pair[0] == "black" or pair[1] == "black", events_captured)))

        return labelling_function, model_metrics, events_captured_filtered
    
    def _on_episode_change(self, previous_episode):
        if self.use_max_episode_length_annealing:
            episode_length_increase = previous_episode * self.max_episode_length_increase_rate
            self.max_episode_length = int(min(self.initial_max_episode_length + episode_length_increase, self.final_max_episode_length))
        super()._on_episode_change(previous_episode)

    def _on_incomplete_episode(self, current_domain_id):
        # if the episode was interrupted, log the learning episode
        self.automaton_learning_episodes[current_domain_id].append(self.current_episode)

    @abstractmethod
    def _choose_action(self, domain_id, task_id, current_state, automaton, current_automaton_state):
        pass

    @abstractmethod
    def _on_performed_step(self, domain_id, task_id, next_state, reward, is_terminal, observations, automaton, current_automaton_state, next_automaton_state, episode_length):
        pass

    @abstractmethod
    def _build_q_functions(self):
        pass

    @abstractmethod
    def _update_q_functions(self, task_id, current_state, action, next_state, is_terminal, observations, observations_changed):
        pass

    @abstractmethod
    def _build_experience_replay_buffers(self):
        pass

    '''
    Greedy Policy Evaluation
    '''
    def _evaluate_greedy_policies(self):
        # we do not want automata to be learned during the evaluation of a policy
        tmp_interleaved_automaton_learning = self.interleaved_automaton_learning
        self.interleaved_automaton_learning = False
        super()._evaluate_greedy_policies()
        self.interleaved_automaton_learning = tmp_interleaved_automaton_learning

    def _set_has_observed_goal_example(self, domain_id, task):
        if task.is_goal_achieved() and not self.has_observed_goal_example[domain_id]:
            self.has_observed_goal_example[domain_id] = True

    def _can_learn_new_automaton(self, domain_id, task):
        self._set_has_observed_goal_example(domain_id, task)
        return not self.wait_for_goal_example or self.has_observed_goal_example[domain_id]

    '''
    Task Management Methods (getting observations)
    '''

    def _get_task_observations_from_env(self, task):
        observations = task.get_observations()
        if self.use_restricted_observables:
            return observations.intersection(task.get_restricted_observables())
        return observations
    
    def _get_task_observations_from_model(self, task, labelling_function, model_metrics, events_captured, state):
        state_tensor = Variable(torch.FloatTensor(state))
        event_vector = labelling_function(state_tensor)
        (event_set, confidence)= self._vector_to_event_with_confidence(event_vector, events_captured, model_metrics)
        # How does restricted_events become the events that we care about for the task at hand without making any major assumptions
        # The confidence score will be affected if we filter out some events when restricted observables are used
        restricted_events = set(filter(self._neaten_event, events_captured))
        if self.use_restricted_observables:
            return (event_set.intersection(restricted_events), confidence)
        return (event_set, confidence)
    
    def _vector_to_event_with_confidence(self, event_vector, events_captured, model_metrics):
        events = set()
        precision_scores = []
        for i in range(len(event_vector)):
            if event_vector[i] > 0.5:
                event, precision = self._convert_event(events_captured[i], model_metrics)
                events.add(event)
                precision_scores.append(precision)
        # Need an indication of no event being observed instead of an empty set because the model will be uncertain about observing no event too
        if not events:
            return (events, model_metrics["precision"]["no_event"])
        else:
            min_precision = utils.min_t_norm_operator(precision_scores)
            return (events, min_precision)
    
    def _convert_event(self, event, model_metrics):
        shortened_event_name = self._neaten_event(event)
        precision = model_metrics["precision"][event]
        return shortened_event_name, precision
        # precision = model_metrics["precision"][event]
        # if event == ('black', 'blue'):
        #     return "b", precision
        # elif event == ('black', 'lime'):
        #     return "g", precision
        # elif event == ('black', 'red'):
        #     return "r", precision
        # elif event == ('black', 'cyan'):
        #     return "c", precision
        # elif event == ('black', 'magenta'):
        #     return "m", precision
        # elif event == ('black', 'yellow'):
        #     return "y", precision

    
    def _neaten_event(self, event):
        return str(event[0][0]) + str(event[1][0])    
    
    '''
    Automata Management Methods (setters, getters, associated rewards)
    '''
    def _get_automaton(self, domain_id):
        return self.automata[domain_id]

    def _get_next_automaton_state(self, automaton, current_automaton_state, observations, observations_changed):
        # automaton has to be navigated with compressed traces if specified (just when a change occurs)
        if (self.ignore_empty_observations and len(observations) == 0) or (self.use_compressed_traces and not observations_changed):
            return current_automaton_state
        return automaton.get_next_state(current_automaton_state, observations)

    def _get_initial_automaton_state_successors(self, domain_id, observations):
        automaton = self._get_automaton(domain_id)
        initial_state = automaton.get_initial_state()
        return self._get_next_automaton_state(automaton, initial_state, observations, True)

    def _set_automata(self, target_automata):
        if self.initial_automaton_mode == "basic":
            self._set_basic_automata()
        elif self.initial_automaton_mode == "load_solution":
            self._load_last_automata_solutions()
        elif self.initial_automaton_mode == "target":
            self.automata = target_automata
        else:
            raise RuntimeError("Error: The initial automaton mode \"%s\" is not recognised." % self.initial_automaton_mode)

    def _load_last_automata_solutions(self):
        self.automata = []

        for i in range(self.num_domains):
            automaton_solution_folder = self.get_automaton_solution_folder(i)
            last_automaton_filename = self._get_last_solution_filename(automaton_solution_folder)
            automaton = self._parse_ilasp_solutions(last_automaton_filename)
            automaton.set_initial_state(ISAAlgorithmBase.INITIAL_STATE_NAME)
            automaton.set_accept_state(ISAAlgorithmBase.ACCEPTING_STATE_NAME)
            automaton.set_reject_state(ISAAlgorithmBase.REJECTING_STATE_NAME)
            self.automata.append(automaton)

    def _get_last_solution_filename(self, automaton_solution_folder):
        automaton_solutions = os.listdir(automaton_solution_folder)
        automaton_solutions.sort(key=lambda k: int(k[:-len(".txt")].split("-")[1]))
        automaton_solutions_path = [os.path.join(automaton_solution_folder, f) for f in automaton_solutions]

        if len(automaton_solutions_path) > 1 and utils.is_file_empty(automaton_solutions_path[-1]):
            return automaton_solutions_path[-2]

        return automaton_solutions_path[-1]

    def _set_basic_automata(self):
        self.automata = []

        for _ in range(self.num_domains):
            # the initial automaton is an automaton that doesn't accept nor reject anything
            automaton = SubgoalAutomaton()
            automaton.add_state(ISAAlgorithmBase.INITIAL_STATE_NAME)
            automaton.set_initial_state(ISAAlgorithmBase.INITIAL_STATE_NAME)
            # automaton.add_state(ISAAlgorithm.ACCEPTING_STATE_NAME)  # DO NOT UNCOMMENT!
            # automaton.add_state(ISAAlgorithm.REJECTING_STATE_NAME)
            # automaton.set_accept_state(ISAAlgorithm.ACCEPTING_STATE_NAME)
            # automaton.set_reject_state(ISAAlgorithm.REJECTING_STATE_NAME)
            self.automata.append(automaton)

    '''
    Automata Learning Methods (example update, task generation/solving/parsing)
    '''
    @abstractmethod
    def _on_automaton_learned(self, domain_id):
        pass

    def _perform_interleaved_automaton_learning(self, task, domain_id, current_automaton_state, observation_history,
                                                compressed_observation_history, events_captured):
        """Updates the set of examples based on the current observed trace. In case the set of example is updated, it
        makes a call to the automata learner. Returns True if a new automaton has been learnt, False otherwise."""
        updated_examples = self._update_examples(task, domain_id, current_automaton_state, observation_history,
                                                 compressed_observation_history)
        if updated_examples:
            if self.debug:
                if self.use_compressed_traces:
                    counterexample = str(compressed_observation_history)
                else:
                    counterexample = str(observation_history)
                print("Updating automaton " + str(domain_id) + "... The counterexample is: " + counterexample)
            self._update_automaton(task, domain_id, events_captured)
            return True  # whether a new automaton has been learnt

        return False

    def _reset_examples(self):
        # there is a set of examples for each domain
        self.goal_examples = [set() for _ in range(self.num_domains)]
        self.dend_examples = [set() for _ in range(self.num_domains)]
        self.inc_examples = [set() for _ in range(self.num_domains)]

    def _update_examples(self, task, domain_id, current_automaton_state, observation_history, compressed_observation_history):
        """Updates the set of examples. Returns True if the set of examples has been updated and False otherwise. Note
        that an update of the set of examples can be forced by setting 'current_automaton_state' to None."""
        automaton = self.automata[domain_id]

        if task.is_terminal():
            if task.is_goal_achieved():
                if current_automaton_state is None or not automaton.is_accept_state(current_automaton_state):
                    print("We have observed a positive counterexample!")
                    # print(observation_history)
                    self._update_example_set(self.goal_examples[domain_id], observation_history, compressed_observation_history)
                    return True
            else:
                if current_automaton_state is None or not automaton.is_reject_state(current_automaton_state):
                    print("We have observed a negative counterexample!")
                    # print(observation_history)
                    self._update_example_set(self.dend_examples[domain_id], observation_history, compressed_observation_history)
                    return True
        else:
            # just update incomplete examples if at least we have one goal or one deadend example (avoid overflowing the
            # set of incomplete unnecessarily)
            if current_automaton_state is None or automaton.is_terminal_state(current_automaton_state):
                print("We have observed an incomplete counterexample!")
                # print(current_automaton_state is None)
                # print(observation_history)
                self._update_example_set(self.inc_examples[domain_id], observation_history, compressed_observation_history)
                return True
        return False  # whether example sets have been updated

    # Do I add the notion of an example weight when adding the observation history to the example set?
    def _update_example_set(self, example_set, observation_history, compressed_observation_history):
        """Updates the a given example set with the corresponding history of observations depending on whether
        compressed traces are used or not to learn the automata. An exception is thrown if a trace is readded."""
        # print(example_set)
        history_to_use = compressed_observation_history if self.use_compressed_traces else observation_history
        event_trace, confidence_scores = zip(*history_to_use)
        history_tuple = tuple(event_trace)
        print(history_tuple)
        # if self.use_compressed_traces:
        #     history_tuple = tuple(compressed_observation_history)
        # else:
        #     history_tuple = tuple(observation_history)

        for example in example_set:
            if history_tuple == example[0]:
                raise RuntimeError("An example that an automaton is currently covered cannot be uncovered afterwards!")
        example_set.add((history_tuple, confidence_scores))
        # if history_tuple not in example_set:
        #     example_set.add((history_tuple, confidence_scores))
        # else:
        #     raise RuntimeError("An example that an automaton is currently covered cannot be uncovered afterwards!")

    def _update_automaton(self, task, domain_id, events_captured):
        self.automaton_counters[domain_id] += 1  # increment the counter of the number of aut. learnt for a domain

        print("Generating ILASP task")
        self._generate_ilasp_task(task, domain_id, events_captured)  # generate the automata learning task

        print("Solving ILASP task")
        solver_success = self._solve_ilasp_task(domain_id)  # run the task solver
        if solver_success:
            ilasp_solution_filename = os.path.join(self.get_automaton_solution_folder(domain_id),
                                                   ISAAlgorithmBase.AUTOMATON_SOLUTION_FILENAME % self.automaton_counters[domain_id])
            candidate_automaton = self._parse_ilasp_solutions(ilasp_solution_filename)

            if candidate_automaton.get_num_states() > 0:
                print("ILASP task solved with satisfiable answer. Plotting automaton")
                # set initial, accepting and rejecting states in the automaton
                candidate_automaton.set_initial_state(ISAAlgorithmBase.INITIAL_STATE_NAME)
                candidate_automaton.set_accept_state(ISAAlgorithmBase.ACCEPTING_STATE_NAME)
                candidate_automaton.set_reject_state(ISAAlgorithmBase.REJECTING_STATE_NAME)
                self.automata[domain_id] = candidate_automaton

                # plot the new automaton
                candidate_automaton.plot(self.get_automaton_plot_folder(domain_id),
                                         ISAAlgorithmBase.AUTOMATON_PLOT_FILENAME % self.automaton_counters[domain_id])

                self._on_automaton_learned(domain_id)
            else:
                # if the task is UNSATISFIABLE, it means the number of states is not enough to cover the examples, so
                # the number of states is incremented by 1 and try again
                print("The task is unsatisfiable with {} states".format(self.num_automaton_states[domain_id]))
                self.num_automaton_states[domain_id] += 1

                if self.debug:
                    print("The number of states in the automaton has been increased to " + str(self.num_automaton_states[domain_id]))
                    print("Updating automaton...")
                # raise RuntimeError("You haven't managed to produce an automaton!")
                self._update_automaton(task, domain_id)
        else:
            raise RuntimeError("Error: Couldn't find an automaton under the specified timeout!")

    # Somewhere in this generation of the ILASP task I need to pass in the fact that the goal, dend and inc examples have weights attached to them?
    def _generate_ilasp_task(self, task, domain_id, events_captured):
        # print(self.get_automaton_task_folder(domain_id))
        utils.mkdir(self.get_automaton_task_folder(domain_id))

        ilasp_task_filename = ISAAlgorithmBase.AUTOMATON_TASK_FILENAME % self.automaton_counters[domain_id]

        # Do the events here also need to be recovered from the labelling function model and not the environment?
        # observables = task.get_observables()
        # if self.use_restricted_observables:
        #     observables = task.get_restricted_observables()

        observables = set(map(self._neaten_event, events_captured))
        # print(observables)

        # the sets of examples are sorted to make sure that ILASP produces the same solution for the same sets (ILASP
        # can produce different hypothesis for the same set of examples but given in different order)
        generate_ilasp_task(self.num_automaton_states[domain_id], ISAAlgorithmBase.ACCEPTING_STATE_NAME,
                            ISAAlgorithmBase.REJECTING_STATE_NAME, observables, sorted(self.goal_examples[domain_id]),
                            sorted(self.dend_examples[domain_id]), sorted(self.inc_examples[domain_id]),
                            self.get_automaton_task_folder(domain_id), ilasp_task_filename, self.symmetry_breaking_method,
                            self.max_disjunction_size, self.learn_acyclic_graph, self.use_compressed_traces,
                            self.avoid_learning_only_negative, self.prioritize_optimal_solutions, self.binary_folder_name)

    def _solve_ilasp_task(self, domain_id):
        utils.mkdir(self.get_automaton_solution_folder(domain_id))

        ilasp_task_filename = os.path.join(self.get_automaton_task_folder(domain_id),
                                           ISAAlgorithmBase.AUTOMATON_TASK_FILENAME % self.automaton_counters[domain_id])

        ilasp_solution_filename = os.path.join(self.get_automaton_solution_folder(domain_id),
                                               ISAAlgorithmBase.AUTOMATON_SOLUTION_FILENAME % self.automaton_counters[domain_id])

        return solve_ilasp_task(ilasp_task_filename, ilasp_solution_filename, timeout=self.ilasp_timeout,
                                version=self.ilasp_version, max_body_literals=self.max_body_literals,
                                binary_folder_name=self.binary_folder_name, compute_minimal=self.ilasp_compute_minimal)

    def _parse_ilasp_solutions(self, last_automaton_filename):
        return ilasp_solution_parser.parse_ilasp_solutions(last_automaton_filename)

    '''
    Logging and Messaging Management Methods
    '''
    def _restore_uncheckpointed_files(self):
        super()._restore_uncheckpointed_files()
        self._remove_uncheckpointed_files()

    def _remove_uncheckpointed_files(self):
        """Removes files which were generated after the last checkpoint."""
        for domain_id in range(self.num_domains):
            counter = self.automaton_counters[domain_id]
            self._remove_uncheckpointed_files_helper(self.get_automaton_task_folder(domain_id), "task-", ".las", counter)
            self._remove_uncheckpointed_files_helper(self.get_automaton_solution_folder(domain_id), "solution-", ".txt", counter)
            self._remove_uncheckpointed_files_helper(self.get_automaton_plot_folder(domain_id), "plot-", ".png", counter)

    def _remove_uncheckpointed_files_helper(self, folder, prefix, extension, automaton_counter):
        if utils.path_exists(folder):
            files_to_remove = [os.path.join(folder, x) for x in os.listdir(folder)
                               if x.startswith(prefix) and int(x[len(prefix):-len(extension)]) > automaton_counter]
            utils.rm_files(files_to_remove)

    def _write_automaton_learning_episodes(self):
        for domain_id in range(self.num_domains):
            utils.mkdir(self.export_folder_names[domain_id])
            with open(self.get_automaton_learning_episodes_file(domain_id), 'w') as f:
                for episode in self.automaton_learning_episodes[domain_id]:
                    f.write(str(episode) + '\n')

    '''
    File Management Methods
    '''
    def get_automaton_task_folders(self):
        return [self.get_automaton_task_folder(domain_id) for domain_id in range(self.num_domains)]

    def get_automaton_task_folder(self, domain_id):
        return os.path.join(self.export_folder_names[domain_id], ISAAlgorithmBase.AUTOMATON_TASK_FOLDER)

    def get_automaton_solution_folders(self):
        return [self.get_automaton_solution_folder(domain_id) for domain_id in range(self.num_domains)]

    def get_automaton_solution_folder(self, domain_id):
        return os.path.join(self.export_folder_names[domain_id], ISAAlgorithmBase.AUTOMATON_SOLUTION_FOLDER)

    def get_automaton_plot_folders(self):
        return [self.get_automaton_plot_folder(domain_id) for domain_id in range(self.num_domains)]

    def get_automaton_plot_folder(self, domain_id):
        return os.path.join(self.export_folder_names[domain_id], ISAAlgorithmBase.AUTOMATON_PLOT_FOLDER)

    def get_automaton_learning_episodes_files(self):
        return [self.get_automaton_learning_episodes_file(domain_id) for domain_id in range(self.num_domains)]

    def get_automaton_learning_episodes_file(self, domain_id):
        return os.path.join(self.export_folder_names[domain_id], ISAAlgorithmBase.AUTOMATON_LEARNING_EPISODES_FILENAME)
