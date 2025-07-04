import os
import sys
import logging
import shutil
import glob
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import json
import copy
import random
from search_space import SearchSpace
from scheme_evaluation_step import SchemeEvaluationStep
from automc.KnowledgeGeneration import KnowledgeGeneration
from automc.KnowledgeModel import KnowledgeModel
from automc.ParetoModel import ParetoModel


class AutoMLOur(object):
      def __init__(self, config_path, task_name, task_info):
          super(AutoMLOur, self).__init__()
  
          # --- logger 初期化ここから ---
          self.logger = logging.getLogger(__name__)
          self.logger.setLevel(logging.INFO)  # 必要に応じて DEBUG に変更も可
  
          ch = logging.StreamHandler()
          ch.setLevel(logging.INFO)
          formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
          ch.setFormatter(formatter)
  
          if not self.logger.hasHandlers():
              self.logger.addHandler(ch)
          # --- logger 初期化ここまで ---
  
          with open(config_path,"r") as f:
              config = json.load(f)
          config['task_name'] = task_name
          config['task_info'] = task_info
  
          self.environment_setting(config)
  
          self.space = SearchSpace
          self.cstartegies = self.get_spacecomponents(self.space)
          self.knowledge_path = "automc"
          if not os.path.exists(self.knowledge_path + '/exp_infos_1.pkl'):
              KnowledgeGeneration(self.space, self.cstartegies, self.knowledge_path, self.logger).main()
          else:
              self.logger.info("* KnowledgeGeneration already finished before")
  
          self.k_model = KnowledgeModel(config, self.cstrategies_path, self.knowledge_path, self.logger).cuda()
          cstrategy_embeddings = self.learn_embeddings()
  
          self.task_info_norm_value = self.k_model.task_info_norm_value
          task_array = self.get_task_array(config)
  
          self.target_compression_rate = float(config["task_name"].split("+")[2])
          self.p_model = ParetoModel(cstrategy_embeddings, task_array, self.target_compression_rate).cuda()
  
          self.evaluator = SchemeEvaluationStep(config, self.logging_path, self.logger)
  
          self.optimal_num = config["our_pmodel_optimal_num"]
          self.update_frequency = config['our_pmodel_update_frequency']
          self.batch_size = config['our_pmodel_batch_size']
          self.avg_ratio = config['our_pmodel_avg_ratio']
          self.update_batch_num = config['our_pmodel_update_batch_num']
          self.optimizer = torch.optim.Adam(self.p_model.parameters(), lr=config["our_pmodel_learning_rate"])
          self.loss = torch.nn.MSELoss()
  
          task_info = self.get_real_taskinfo(config["task_info"])
          next_cstrategies = [i for i in range(1, len(self.cstartegies)+1)]
          self.history = [
              {
                  "pre_sequences": [0],
                  "selected_next_cstrategy": [],
                  "valid_unselected_next_cstrategy": list(next_cstrategies),
                  "score_info": [task_info["top1_acc"], 0, 0, task_info["parameter_amount"], task_info["flops_amount"]],
                  "pre_info": [None,None,None,None,None],
                  "source_index": 0,
                  "pareto_score_value": [0,0],
                  "finished_compression_rate": 0,
                  "finished_finetune_rate": 0,
                  "valid": True
              }
          ]
          self.pareto_history = [
              {
                  "source_index": 0,
                  "pareto_score_value": [0,0],
              }
          ]
          self.avg_pareto_history = [
              {
                  "source_index": 0,
                  "avg_pareto_score_value": [0,0],
              }
          ]
          self.p_model_data = []
  
          self.automl_search_time_s = float(config['automl_search_time(h)'])*60*60
  
          self.codes = 0
          self.valid_codes = 0
          self.steps = 0
          self.valid_steps = 0

      def create_exp_dir(self, path, scripts_to_save=None):
        if not os.path.exists(path):
          os.mkdir(path)

        if scripts_to_save is not None:
          os.mkdir(os.path.join(path, 'scripts'))
          for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
        return
      
      def environment_setting(self, config):
        logging_dir = config["logging_dir"]
        if not os.path.exists(logging_dir):
          os.mkdir(logging_dir)
        logging_path = logging_dir + '/AutoML_Our-Run-{}'.format(time.strftime("%Y%m%d-%H%M%S"))
        self.create_exp_dir(logging_path, scripts_to_save=glob.glob('*.py'))
        self.logging_path = logging_path
        
        log_format = '%(asctime)s %(message)s'
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(logging_path, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(fh)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(stream_handler)

        if not torch.cuda.is_available():
          self.logger.info('* no gpu device available')
          np.random.seed(config["seed"])
          self.logger.info("* config = %s", str(config))
          sys.exit(1)

        np.random.seed(config["seed"])
        #torch.cuda.set_device(config["gpu_device"])
        cudnn.benchmark = True
        torch.manual_seed(config["seed"])
        cudnn.enabled = True
        torch.cuda.manual_seed(config["seed"])
        self.logger.info("* config = %s", str(config))
        return

      def get_sub_spacecomponents(self, calg, hpo_dict, hpo_name):
        sub_cstartegies = []
        if len(hpo_name) == 1:
          for hpo_value in hpo_dict[hpo_name[0]]:
            content = {}
            content[hpo_name[0]] = hpo_value
            sub_cstartegies.append([calg, content])
          return sub_cstartegies
        contents = self.get_sub_spacecomponents(calg, hpo_dict, list(hpo_name[1:]))
        for hpo_value in hpo_dict[hpo_name[0]]:
          for i in range(len(contents)):
            content = copy.deepcopy(contents[i][1])
            content[hpo_name[0]] = hpo_value
            sub_cstartegies.append([calg, content])
        return sub_cstartegies
      def get_spacecomponents(self, space):
        self.logger.info('* get_spacecomponents now')
        if os.path.exists("cstartegies.txt"):
          with open("cstartegies.txt", "r") as f:
            cstartegies = eval(f.readline())
          self.logger.info('* %d cstrategies in all' % len(cstartegies))
          self.cstrategies_path = "cstartegies.txt"
          return cstartegies
        cstartegies = []
        for calg in space.keys():
          calg_hpo_dict = space[calg]
          calg_hpo_name = list(calg_hpo_dict.keys())
          sub_cstartegies = self.get_sub_spacecomponents(calg, calg_hpo_dict, calg_hpo_name)
          cstartegies.extend(sub_cstartegies)
          self.logger.info('* %d cstrategies in calg: %s', len(sub_cstartegies), calg)
        self.logger.info('* %d cstrategies in all', len(cstartegies))
        with open("cstartegies.txt", "w") as f:
          f.write(str(cstartegies))
        self.cstrategies_path = "cstartegies.txt"
        return cstartegies

      def learn_embeddings(self):
        self.k_model.train()
        cstrategy_embeddings = self.k_model.main()
        return cstrategy_embeddings

      
      def get_real_taskinfo(self, task_info):
            return task_info

      
      def get_task_array(self, config):
          # config["task_info"] はすでに辞書なので、そのまま使ってOK
          task_info = config["task_info"]
      
          task_array = np.array([
              task_info["class_num"],
              task_info["image_size"],
              task_info["image_channel"],
              task_info["avg_data_num_per_class"],
              task_info["top1_acc"],
              task_info["parameter_amount"],
              task_info["flops_amount"]
          ])
          
          task_array = task_array / self.task_info_norm_value
          self.logger.info('* task_array: %s, task_info_norm_value: %s', str(task_array), str(self.task_info_norm_value))
          return task_array


      def get_next_candidates(self):
          self.logger.info('* get_next_candidates begin')
          next_candidates = []
          next_cstrategies = [i for i in range(1, len(self.cstartegies) + 1)]
      
          # ----- sample from pareto_history -----
          pareto_num = self.batch_size - int(self.batch_size * self.avg_ratio)
          self.logger.info('len(pareto_history): %d', len(self.pareto_history))
      
          if len(self.pareto_history) > 0:
              self.pareto_history = sorted(
                  self.pareto_history,
                  key=lambda info: info["pareto_score_value"][0],
                  reverse=True
              )
      
              pareto_history_valid = []
              for info in self.pareto_history:
                  source_idx = info["source_index"]
                  if 0 <= source_idx < len(self.history) and self.history[source_idx]["valid"]:
                      pareto_history_valid.append(info)
      
              if len(pareto_history_valid) > 0:
                  top = max(int(len(pareto_history_valid) * 0.2), 1)
                  try_num = 0
                  while pareto_num != 0 and try_num < 1000:
                      try:
                          # validな範囲で選択
                          if random.random() <= 0.5:
                              subset = pareto_history_valid[:top]
                          else:
                              subset = pareto_history_valid[top:]
                          if len(subset) == 0:
                              break
      
                          info = random.choice(subset)
                          source_idx = info["source_index"]
      
                          # source_idx の安全確認
                          if not (0 <= source_idx < len(self.history)):
                              self.logger.warning(f"[pareto] source_idx {source_idx} out of range")
                              try_num += 1
                              continue
      
                          h = self.history[source_idx]
                          valid_next = h.get("valid_unselected_next_cstrategy", [])
                          if not valid_next:
                              self.logger.warning(f"[pareto] Empty next_cstrategy @ {source_idx}")
                              try_num += 1
                              continue
      
                          candidate = {
                              "pre_sequences": list(h.get("pre_sequences", [])),
                              "next_cstrategy": [random.choice(valid_next)],
                              "score_info": h.get("score_info"),
                              "pre_info": h.get("pre_info"),
                              "source_index": source_idx,
                              "predicted_step_scores": [None, None],
                              "real_step_scores": [None, None]
                          }
      
                          if candidate not in next_candidates:
                              next_candidates.append(candidate)
                              pareto_num -= 1
                      except Exception as e:
                          self.logger.error(f"[pareto] Exception: {e}")
                      try_num += 1
              else:
                  self.logger.warning("No valid entries in pareto_history.")
          else:
              self.logger.warning("pareto_history is empty.")
      
          self.logger.info('* get_next_candidates from pareto_history: %d', len(next_candidates))
      
          # ----- sample from avg_pareto_history -----
          avg_pareto_num = self.batch_size - len(next_candidates)
          self.logger.info('len(avg_pareto_history): %d', len(self.avg_pareto_history))
      
          if len(self.avg_pareto_history) > 0:
              self.avg_pareto_history = sorted(
                  self.avg_pareto_history,
                  key=lambda info: info["avg_pareto_score_value"][0],
                  reverse=True
              )
      
              avg_pareto_history_valid = []
              for info in self.avg_pareto_history:
                  source_idx = info["source_index"]
                  if 0 <= source_idx < len(self.history) and self.history[source_idx]["valid"]:
                      avg_pareto_history_valid.append(info)
      
              if len(avg_pareto_history_valid) > 0:
                  top = max(int(len(avg_pareto_history_valid) * 0.2), 1)
                  try_num = 0
                  while avg_pareto_num != 0 and try_num < 1000:
                      try:
                          if random.random() <= 0.5:
                              subset = avg_pareto_history_valid[:top]
                          else:
                              subset = avg_pareto_history_valid[top:]
                          if len(subset) == 0:
                              break
      
                          info = random.choice(subset)
                          source_idx = info["source_index"]
      
                          if not (0 <= source_idx < len(self.history)):
                              self.logger.warning(f"[avg_pareto] source_idx {source_idx} out of range")
                              try_num += 1
                              continue
      
                          h = self.history[source_idx]
                          valid_next = h.get("valid_unselected_next_cstrategy", [])
                          if not valid_next:
                              self.logger.warning(f"[avg_pareto] Empty next_cstrategy @ {source_idx}")
                              try_num += 1
                              continue
      
                          candidate = {
                              "pre_sequences": list(h.get("pre_sequences", [])),
                              "next_cstrategy": [random.choice(valid_next)],
                              "score_info": h.get("score_info"),
                              "pre_info": h.get("pre_info"),
                              "source_index": source_idx,
                              "predicted_step_scores": [None, None],
                              "real_step_scores": [None, None]
                          }
      
                          if candidate not in next_candidates:
                              next_candidates.append(candidate)
                              avg_pareto_num -= 1
                      except Exception as e:
                          self.logger.error(f"[avg_pareto] Exception: {e}")
                      try_num += 1
              else:
                  self.logger.warning("No valid entries in avg_pareto_history.")
          else:
              self.logger.warning("avg_pareto_history is empty.")
      
          self.logger.info('* get_next_candidates from avg_pareto_history: %d', len(next_candidates))
      
          # ----- sample from history -----
          left_num = self.batch_size - len(next_candidates)
          history_valid = [h for h in self.history if h.get("valid", False)]
      
          try_num = 0
          while left_num > 0 and try_num < 1000:
              try:
                  if len(history_valid) == 0:
                      self.logger.warning("No valid entries in history to sample.")
                      break
      
                  h = random.choice(history_valid)
                  valid_next = h.get("valid_unselected_next_cstrategy", [])
                  if not valid_next:
                      self.logger.warning(f"[history] Empty next_cstrategy @ {h.get('source_index', 'unknown')}")
                      try_num += 1
                      continue
      
                  candidate = {
                      "pre_sequences": list(h.get("pre_sequences", [])),
                      "next_cstrategy": [random.choice(valid_next)],
                      "score_info": h.get("score_info"),
                      "pre_info": h.get("pre_info"),
                      "source_index": h.get("source_index"),
                      "predicted_step_scores": [None, None],
                      "real_step_scores": [None, None]
                  }
      
                  if candidate not in next_candidates:
                      next_candidates.append(candidate)
                      left_num -= 1
              except Exception as e:
                  self.logger.error(f"[history] Exception: {e}")
              try_num += 1
      
          self.logger.info('* get_next_candidates from history: %d', len(next_candidates))
          self.logger.info('* get_next_candidates finished')
          return next_candidates



      def get_best_candidate(self, next_candidates):
          self.logger.info('* get_best_candidate begin')
      
          if not next_candidates:
              self.logger.warning('No next_candidates available.')
              return [], []
      
          pre_sequences, next_cstrategies, history_scheme_scores = [], [], []
          for i, candidate in enumerate(next_candidates):
              pre_sequences.append(candidate["pre_sequences"])
              next_cstrategies.append(candidate["next_cstrategy"])
              score_info = candidate["score_info"]
              scheme_score = [score_info[3], score_info[0], score_info[1]]  # parameter, acc, compression_rate
              history_scheme_scores.append(scheme_score)
      
          pre_sequences = np.array(pre_sequences)
          next_cstrategies = np.array(next_cstrategies)
          history_scheme_scores = np.array(history_scheme_scores)
      
          self.logger.info(f'pre_sequences.shape={pre_sequences.shape}')
          self.logger.info(f'next_cstrategies.shape={next_cstrategies.shape}')
          self.logger.info(f'history_scheme_scores.shape={history_scheme_scores.shape}')
      
          try:
              optimal_scheme_indexs, predicted_optimal_step_scores = self.p_model.main(
                  pre_sequences, next_cstrategies, history_scheme_scores, self.optimal_num
              )
      
              # NaNチェック
              if predicted_optimal_step_scores is None:
                  raise ValueError("predicted_optimal_step_scores is None")
      
              if torch.isnan(predicted_optimal_step_scores).any():
                  self.logger.error("❗ predicted_optimal_step_scores に NaN が含まれています")
                  self.logger.error("値: %s", str(predicted_optimal_step_scores))
                  raise ValueError("predicted_optimal_step_scores に NaN")
          except Exception as e:
              self.logger.error(f'Exception in p_model.main or NaN check: {str(e)}')
              return [], []
      
          best_candidates = []
          for i, index in enumerate(optimal_scheme_indexs):
              try:
                  if index >= len(next_candidates):
                      self.logger.warning(f'Index out of range in best_candidates: {index}')
                      continue
      
                  best_candidate = copy.deepcopy(next_candidates[index])  # 念のため deepcopy
                  best_candidate["predicted_step_scores"] = predicted_optimal_step_scores[i].data.cpu().numpy()
                  best_candidates.append(best_candidate)
              except Exception as e:
                  self.logger.error(f'Exception while building best_candidates: {str(e)}')
                  continue
      
          self.logger.info('* get_best_candidate finished')
          self.logger.info(f'* best_candidates count: {len(best_candidates)}')
          return best_candidates, predicted_optimal_step_scores


      def pareto_opt_tell(self, score1, score2):
          # score1, score2 はリストやタプルなど比較対象のスコア
          # 例：
          # score1 = [x1, y1]
          # score2 = [x2, y2]
          # ここで pareto dominance の判定を行う実装が入る
      
          # 例示：score1 が score2 を支配していれば True, 逆なら False, どちらもでなければ None など
          # （この判定ロジックは実装による）
          
          if (score1[0] >= score2[0] and score1[1] >= score2[1]) and (score1[0] > score2[0] or score1[1] > score2[1]):
              return True
          elif (score2[0] >= score1[0] and score2[1] >= score1[1]) and (score2[0] > score1[0] or score2[1] > score1[1]):
              return False
          else:
              return None




      def evaluate_best_candidate(self, best_candidates):
          filtered_candidates = []
          for i, c in enumerate(best_candidates):
              # NaNチェック：NaNがあればログ警告し、その候補はスキップ
              if "real_step_scores" in c:
                  if any([isinstance(s, float) and np.isnan(s) for s in c["real_step_scores"]]):
                      self.logger.warning("❗ best_candidate[%d] の real_step_scores に NaN が含まれています: %s", i, c["real_step_scores"])
                      continue
      
              self.codes += 1
              self.steps += 1
              step_code_index = c["next_cstrategy"][0]
              step_code = self.cstartegies[step_code_index - 1]
      
              self.logger.info('@ codes: %s, steps: %s', self.codes, self.steps)
              self.logger.info('@ original step_code: %s', step_code)
      
              self.valid_codes += 1
              self.valid_steps += 1
              self.logger.info('@ valid: True')
              self.logger.info('@ valid_codes: %s, valid_steps: %s', self.valid_codes, self.valid_steps)
              self.logger.info('@ adjusted step_code: %s', step_code)
      
              scheme_code = [self.cstartegies[idx - 1] for idx in c["pre_sequences"][1:]] + [step_code]
              self.logger.info('@ scheme_code: %s', scheme_code)
      
              pre_model_dir, pre_parameter_remain_rate, pre_flops_remain_rate, pre_acc_top1_rate, pre_acc_top5_rate = c["pre_info"]
              step_info, score_info, new_pre_info, table_infos = self.evaluator.main(
                  step_code, pre_model_dir, pre_parameter_remain_rate, pre_flops_remain_rate, pre_acc_top1_rate, pre_acc_top5_rate
              )
      
              # real_step_scoresを最新化
              c["real_step_scores"] = [step_info[0], step_info[2]]  # step_parameter_decreased_ratio, step_acc_increased_ratio
              scheme_score = list(score_info)
      
              self.logger.info('@ scheme_score [step_score, compression_rate, flops_decreased_rate, parameter_amount, flops_amount]: %s', score_info)
              self.logger.info('@ table_infos: %s', table_infos)
      
              self.update_history(c, score_info, new_pre_info, step_code, step_code_index)
      
              # 最適結果の更新
              if self.score_opt_scheme["scheme_score"] is None or (scheme_score[0] > self.score_opt_scheme["scheme_score"][0] and scheme_score[1] >= self.target_compression_rate):
                  self.score_opt_scheme = {
                      "codes/valid_codes": [self.codes, self.valid_codes],
                      "steps/valid_steps": [self.steps, self.valid_steps],
                      "scheme_code": scheme_code,
                      "scheme_score": scheme_score,
                      "table_infos": table_infos
                  }
      
              # パレートフロントの更新
              if scheme_score[1] >= self.target_compression_rate:
                  pareto_front, removed = True, []
                  for pf_scheme in self.pareto_front_schemes_info:
                      if self.pareto_opt_tell(scheme_score, pf_scheme["scheme_score"]):
                          removed.append(pf_scheme)
                      elif self.pareto_opt_tell(pcheme_score, pf_scheme["scheme_score"]) == False:
                          pareto_front = False
                  if pareto_front:
                      self.pareto_front_schemes_info.append({
                          "codes/valid_codes": [self.codes, self.valid_codes],
                          "steps/valid_steps": [self.steps, self.valid_steps],
                          "scheme_code": scheme_code,
                          "scheme_score": scheme_score,
                          "table_infos": table_infos
                      })
                  for item in removed:
                      self.pareto_front_schemes_info.remove(item)
      
              self.logger.info('@ score_opt_scheme: %s', self.score_opt_scheme)
              self.logger.info('@ pareto_front_schemes_info count: %d', len(self.pareto_front_schemes_info))
      
              search_time = time.time() - self.start_time
              self.logger.info('@ automl_search_time (seconds): %.4f, (hours): %.4f', search_time, search_time / 3600.0)
              self.logger.info('@ automl_search_time left (seconds): %.4f, (hours): %.4f',
                              self.automl_search_time_s - search_time,
                              (self.automl_search_time_s - search_time) / 3600.0)
              self.logger.info('@ average time (seconds)/(minutes) per valid_code: %.4f / %.4f',
                              search_time / max(self.valid_codes, 1),
                              search_time / max(self.valid_codes, 1) / 60.0)
      
              filtered_candidates.append(c)
      
          return filtered_candidates


      def get_valid_unselected_next_cstrategy(self, best_candidate_info):
        '''
        Requirements: 
          scheme_code's finished_compression_rate >= 1.0 
          scheme_code's finished_compression_rate * self.target_compression_rate < 0.85
          scheme_code's finished_finetune_rate < 4.0 
        '''
        valid_unselected_next_cstrategy = []
        if best_candidate_info["valid"]:
          pre_sequences = best_candidate_info["pre_sequences"]
          finished_compression_rate = best_candidate_info["finished_compression_rate"]
          finished_finetune_rate = best_candidate_info["finished_finetune_rate"]
          
          for i in range(1, len(self.cstartegies)+1):
            calg, calg_hpo = self.cstartegies[i-1]
            if calg == "prune_C7" and len(pre_sequences) > 1:
              continue
              
            c_rate = finished_compression_rate + calg_hpo["HP2"] * self.target_compression_rate
            if "HP10" in list(calg_hpo.keys()):
              name = "HP10"
            else:
              name = "HP1"
            fe_rate = finished_finetune_rate + calg_hpo[name]
            
            if fe_rate <= 4.0 and c_rate <= 0.85:
              if calg == "prune_C5" and c_rate < self.target_compression_rate:
                continue
              valid_unselected_next_cstrategy.append(i)
        return valid_unselected_next_cstrategy


      def update_history(self, best_candidate, score_info, new_pre_info, step_code, step_code_index):
          # update history, pareto_history, avg_pareto_history
          # update history
          source_index = best_candidate["source_index"]
          #self.logger.info('\n\n\t\tbest_candidate: %s', str(best_candidate))
          #self.logger.info('\t\tsource_index: %d', source_index)
          #self.logger.info('\t\thistory[source_index][valid_unselected_next_cstrategy]: %s', str(self.history[source_index]["valid_unselected_next_cstrategy"]))
          #self.logger.info('\t\thistory[source_index][selected_next_cstrategy]: %s', str(self.history[source_index]["selected_next_cstrategy"]))
          #self.logger.info('\t\tstep_code_index: %d', step_code_index)
          #self.logger.info('\t\tstep_code: %s', str(step_code))
      
          self.history[source_index]["selected_next_cstrategy"].append(step_code_index)
          self.history[source_index]["valid_unselected_next_cstrategy"].remove(step_code_index)
          if len(self.history[source_index]["valid_unselected_next_cstrategy"]) == 0:
              self.history[source_index]["valid"] = False
      
          pre_sequences = copy.deepcopy(best_candidate["pre_sequences"])
          pre_sequences.append(step_code_index)
      
          step_score, compression_rate, flops_decreased_rate, parameter_amount, flops_amount = score_info
          left_compression_rate = max(self.target_compression_rate - compression_rate, 0) + 1.0
          pareto_score_value = [step_score / left_compression_rate, compression_rate / left_compression_rate]
      
          real_socre_info = [float(step_score), float(compression_rate), float(flops_decreased_rate), float(parameter_amount[:-1]), float(flops_amount[:-1]) * 1000 / 2]
      
          if "HP10" in step_code[1].keys():
              new_finished_finetune_rate = step_code[1]["HP10"]
          else:
              new_finished_finetune_rate = step_code[1]["HP1"]
          finished_finetune_rate = self.history[source_index]["finished_finetune_rate"] + new_finished_finetune_rate
          if float(compression_rate) <= 0.85 and float(finished_finetune_rate) + 0.2 <= 4.0:
              valid = True
              if score_info == [0, 0, 0, '0M', '0G']:
                  valid = False
              elif self.cstartegies[step_code_index - 1][0] in ["prune_C5", "prune_C7"]:
                  valid = False
          else:
              valid = False
          best_candidate_info = {
              "pre_sequences": pre_sequences,
              "selected_next_cstrategy": [],
              "valid_unselected_next_cstrategy": [],
              "score_info": list(real_socre_info),
              "pre_info": list(new_pre_info),
              "source_index": len(self.history),
              "pareto_score_value": list(pareto_score_value),
              "finished_compression_rate": float(compression_rate),  # required: self.target_compression_rate * finished_compression_rate <= 0.85
              "finished_finetune_rate": float(finished_finetune_rate),  # required: finished_finetune_rate <= 4.0
              "valid": valid
          }
          valid_unselected_next_cstrategy = self.get_valid_unselected_next_cstrategy(best_candidate_info)
          best_candidate_info["valid_unselected_next_cstrategy"] = list(valid_unselected_next_cstrategy)
          if len(best_candidate_info["valid_unselected_next_cstrategy"]) == 0:
              best_candidate_info["valid"] = False
          self.history.append(best_candidate_info)
      
          # update pareto_history
          pareto_front, removed = True, []
          for j in range(len(self.pareto_history)):
              pareto_score_value_j = self.pareto_history[j]["pareto_score_value"]
              if self.pareto_opt_tell(pareto_score_value, pareto_score_value_j) == True:
                  removed.append(self.pareto_history[j])
              elif self.pareto_opt_tell(pareto_score_value, pareto_score_value_j) == False:
                  pareto_front = False
      
          if pareto_front:
              best_candidate_pareto_info = {
                  "source_index": best_candidate_info["source_index"],
                  "pareto_score_value": list(pareto_score_value)
              }
              self.pareto_history.append(best_candidate_pareto_info)
          for item in removed:
              self.pareto_history.remove(item)
      
          # update avg_pareto_history
          avg_pareto_score_value = [pareto_score_value[0] / len(pre_sequences), pareto_score_value[1] / len(pre_sequences)]
      
          pareto_front, removed = True, []
          for j in range(len(self.avg_pareto_history)):
              avg_pareto_score_value_j = self.avg_pareto_history[j]["avg_pareto_score_value"]
              if self.pareto_opt_tell(avg_pareto_score_value, avg_pareto_score_value_j) == True:
                  removed.append(self.avg_pareto_history[j])
              elif self.pareto_opt_tell(avg_pareto_score_value, avg_pareto_score_value_j) == False:
                  pareto_front = False
      
          if pareto_front:
              best_candidate_avg_pareto_info = {
                  "source_index": best_candidate_info["source_index"],
                  "avg_pareto_score_value": list(avg_pareto_score_value)
              }
              self.avg_pareto_history.append(best_candidate_avg_pareto_info)
          for item in removed:
              self.avg_pareto_history.remove(item)
          return

      def get_history_data_batch(self):
        if len(self.p_model_data) > self.batch_size:
          history_data = random.sample(self.p_model_data, self.batch_size)
        else:
          history_data = list(self.p_model_data)
        pre_sequences, next_cstrategies, history_scheme_scores = [], [], []
        for i in range(len(history_data)):
          data = history_data[i]
          #self.logger.info('\t@ %d, data: %s', i, str(data))
          pre_sequences.append(data["pre_sequences"])
          next_cstrategies.append(data["next_cstrategy"])
          score_info = data["score_info"]
          scheme_score = [score_info[3], score_info[0], score_info[1]] # parameter, acc, compression_rate
          history_scheme_scores.append(scheme_score)

        pre_sequences = np.array(pre_sequences)
        next_cstrategies = np.array(next_cstrategies)
        history_scheme_scores = np.array(history_scheme_scores)

        #self.logger.info('@ history_data: %s', str(history_data))
        #self.logger.info('@ pre_sequences: %s', str(pre_sequences))
        #self.logger.info('@ next_cstrategies: %s', str(next_cstrategies))
        #self.logger.info('@ history_scheme_scores: %s', str(history_scheme_scores))

        predicted_optimal_step_scores = self.p_model.main(pre_sequences, next_cstrategies, history_scheme_scores, -1)

        # predicted_optimal_step_scores: parameter_decreased_ratio, acc_increased_ratio
        ground_truth = []
        for i in range(len(history_data)):
          ground_truth.append(history_data[i]["real_step_scores"])
        ground_truth = torch.FloatTensor(ground_truth).reshape(-1,2).cuda()
        return ground_truth, predicted_optimal_step_scores

      def update_p_model(self, best_candidates, predicted_optimal_step_scores):
          # NaN除去付き ground_truth 作成
          ground_truth = []
          valid_indices = []
          for i in range(len(best_candidates)):
              score = best_candidates[i]["real_step_scores"]
              if any([np.isnan(s) for s in score if isinstance(s, float)]):
                  self.logger.warning("❗ p_model update skipped due to NaN in best_candidates[%d]: %s", i, score)
                  continue
              ground_truth.append(score)
              valid_indices.append(i)
      
          if len(ground_truth) == 0:
              self.logger.warning("⚠️ No valid ground_truth to update p_model.")
              return
      
          ground_truth = torch.FloatTensor(ground_truth).reshape(-1, 2).cuda()
          predicted_optimal_step_scores = predicted_optimal_step_scores[valid_indices]
      
          self.optimizer.zero_grad()
          loss_value = self.loss(predicted_optimal_step_scores, ground_truth) 
          loss_value.backward()
          self.optimizer.step()
          self.logger.info('@ loss_value 1 of p_model: %.2f', loss_value.item())
      
          self.logger.info('@ number of p_model_data obtained before: %d', len(self.p_model_data))
          self.logger.info('@ p_model_data: %s', str(self.p_model_data))
      
          if len(self.p_model_data) > 0:
              for i in range(self.update_batch_num):
                  ground_truth_batch, predicted_optimal_step_scores_batch = self.get_history_data_batch()
      
                  self.optimizer.zero_grad()
                  loss_value_batch = self.loss(predicted_optimal_step_scores_batch, ground_truth_batch) 
                  loss_value_batch.backward()
                  self.optimizer.step()
      
                  self.logger.info('@ update batch_num of p_model: %d', i)
                  self.logger.info('@ loss_value 2 of p_model: %.2f', loss_value_batch.item())
      
          self.p_model_data.extend(copy.deepcopy([best_candidates[i] for i in valid_indices]))
          return

      def safe_parse_value(s, unit='M', logger=None):
          """
          Parses a string like '12.3M' or '1.5G' into a float value in MB.
          If 'G' is specified, it converts GB to MB (using 1G = 500M as in original logic).
          If parsing fails, returns 0.0 and logs the warning if logger is provided.
          """
          try:
              if not isinstance(s, str):
                  raise ValueError(f"Input is not a string: {s}")
              
              value = float(s.strip().replace(unit, ''))
              if unit == 'M':
                  return value
              elif unit == 'G':
                  return value * 500  # original logic: *1000 / 2
              else:
                  raise ValueError(f"Unsupported unit: {unit}")
          except Exception as e:
              if logger:
                  logger.warning("❗ Failed to parse value '%s': %s", s, str(e))
              return 0.0


      def main(self):
          self.pareto_front_schemes_info = []
          self.score_opt_scheme = {"codes/valid_codes": [None,None], "steps/valid_steps": [None,None], "scheme_code": None, "scheme_score": None}
      
          self.start_time = time.time()
          search_time = 0.0
          iteration = 0
          self.p_model.train()
          while search_time < self.automl_search_time_s:
              self.iteration = iteration
              self.logger.info('@ iteration: %d', iteration)
      
              next_candidates = self.get_next_candidates()
              best_candidates, predicted_optimal_step_scores = self.get_best_candidate(next_candidates)
              best_candidates = self.evaluate_best_candidate(best_candidates)
      
              iteration += 1
              if iteration % self.update_frequency == 0:
                  self.logger.info('@ update p_model')
                  self.update_p_model(best_candidates, predicted_optimal_step_scores)
      
              search_time = time.time() - self.start_time  # <= これを忘れずに！！
      
          self.logger.info('@ Final pareto_front_schemes_info: %s', str(self.pareto_front_schemes_info))
          # ... (省略)
          return


'''
if __name__ == '__main__':
  config_path = "config.json"
  task_name = "resnet56+mini_cifar10+0.3"
  task_info = "{'class_num': 10, 'image_size': 32, 'image_channel': 3, 'avg_data_num_per_class': 600, 'top1_acc': 0.6763, 'top5_acc': 0, 'parameter_amount': 0.284, 'flops_amount': 43.42}"
  nas_obj = AutoMLOur(config_path, task_name, task_info)
  nas_obj.main()
'''
