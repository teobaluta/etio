library(bnlearn)
library(Rgraphviz)
library(graph)
library(comprehenr)

get_features = function(attack, weight_decay, with_proxy_pred_vec=FALSE) {
  if (attack == "Top3MLLeakLAcc" || attack == "Top3MLLeakAcc" || grepl("InfAcc", attack, fixed = TRUE)) {
    features = c("TrainAcc", "TestAcc", "AccDiff",
                 "TrainLoss", "TestLoss",
                 "LossDiff",
                 "TrainVar", "TestVar", "TrainBias2", "TestBias2",
                 "NumParams", "CentroidDistance.sorted_3.",
                 "TrainSize") #, "WeightDecay")
  } else {
    features = c("TrainAcc", "TestAcc", "AccDiff",
                 "TrainLoss", "TestLoss",
                 "LossDiff",
                 "TrainVar", "TestVar", "TrainBias2", "TestBias2",
                 "NumParams", "CentroidDistance.origin.",
                 "TrainSize") #, "WeightDecay")
  }
  if (with_proxy_pred_vec == TRUE) {
    features = append(features, "U1", "U2")
  }
  if (weight_decay <= 0)
    features = append(features, "WeightDecay")
  return(features)
}

get_cv_loss_score = function(discretize_opt) {
  if (discretize_opt == "discretize") {
    loss1 = "pred"
    loss2 = NA
  } else if (discretize_opt == "hybrid") {
    loss1 = "cor-lw-cg"
    loss2 = "mse-lw-cg"
  # This is the case where we limit our net to a Gaussian Bayesian network
  } else if (discretize_opt == "cont") {
    loss1 = "cor"
    loss2 = "mse"
  } else {
    quit("Unrecognized discretize_opt")
  }

  return(c(loss1, loss2))
}

average_effect_on_attack = function(fitted, attack, cause, lvl_low, lvl_high) {
  # Estimate the conditional probability of Attack|Cause<>condition_lvl
  # Work-around
  # https://stackoverflow.com/questions/62124076/using-cpquery-function-for-several-pairs-from-dataset
  query_txt_0 = sprintf("cpdist(fitted, nodes=c('%s','%s'),n=10^4, evidence=(%s < %f))",
                        attack, cause, cause, lvl_low)
  print(query_txt_0)
  sim_0 = eval(parse(text=query_txt_0))
  query_txt_1 = sprintf("cpdist(fitted, nodes=c('%s','%s'), n=10^4, evidence=(%s >= %f))",
                        attack, cause, cause, lvl_high)
  print(query_txt_1)
  sim_1 = eval(parse(text=query_txt_1))
  # If we use cpdist, it returns a dataframe containing the samples generated from the
  # conditional distribution of the nodes conditioned on the evidence()
  cate = mean(sim_1[[attack]]) - mean(sim_0[[attack]])
  # If cpquery, then we directly get a conditional probability
  #cate = sim_1 - sim_0
  return(cate)
}

estimate_effect = function(fitted, features, dataset, attack) {
  results = data.frame(matrix(ncol = 5, nrow = 0))
  for (feat in features) {
    if (feat != attack) {
      lvl_low = min(dataset[[feat]])
      lvl_high = max(dataset[[feat]])
      cate = average_effect_on_attack(fitted, attack, feat, lvl_low, lvl_high)
      results[nrow(results)+1,] = c(attack, feat, lvl_low, lvl_high, cate)
    }
    print(sprintf("CATE of E[%s|%s < %f] - E[%s|%s >= %f]=%f", attack, feat, lvl_low, attack, feat, lvl_high, cate))
  }
  return(results)
}

get_root_nodes = function(features) {
  root_nodes = c()
  potential_root_nodes = c("NumParams", "WeightDecay", "TrainSize", "Scheduler.", "EpochNum")
  for (node in potential_root_nodes) {
    if (node %in% features) {
      root_nodes = c(root_nodes, node)
    }
  }

  return(root_nodes)
}

create_wl_bl = function(features, attack) {
  wl = matrix(c("TrainAcc", "AccDiff",
                "TrainLoss", "LossDiff",
                "TrainLoss", "TrainBias2",
                "TrainVar", "TrainBias2",
                "TestLoss", "LossDiff",
                "TestLoss", "TestBias2",
                "TestVar", "TestBias2",
                "TestAcc", "AccDiff"
                ),
              ncol=2, byrow=TRUE, dimnames = list(NULL, c("from", "to")))
  # These are the root nodes, they have no parents
  root_nodes = get_root_nodes(features)

  # Constraint: No edge from the attack node to any of the other nodes
  bl = tiers2blacklist(list(features, attack))

  # Constraint: TrainVar has no incoming edge from itself and all other nodes except root nodes
  feat_excl = to_vec(for (x in features) if (x != "TrainVar" && !(x %in% root_nodes)) x)
  bl_extra = tiers2blacklist(list("TrainVar", feat_excl))
  bl = rbind(bl, bl_extra)

  # Constraint: TrainBias has no incoming edges except root nodes, TrainBias2, TrainVar and TrainLoss
  feat_excl = to_vec(for (x in features) if (x != "TrainBias2" && x != "TrainVar" && x != "TrainLoss" &&
                                            !(x %in% root_nodes)) x)
  bl_extra = tiers2blacklist(list("TrainBias2", feat_excl))
  bl = rbind(bl, bl_extra)

  # Constraint: TrainLoss has no incoming edge except root nodes
  feat_excl = to_vec(for (x in features) if (x != "TrainLoss" && !(x %in% root_nodes)) x)
  bl_extra = tiers2blacklist(list("TrainLoss", feat_excl))
  bl = rbind(bl, bl_extra)

  # Constraint: TrainAcc has no incoming edges except root nodes
  feat_excl = to_vec(for (x in features) if (x != "TrainAcc" && !(x %in% root_nodes)) x)
  bl_extra = tiers2blacklist(list("TrainAcc", feat_excl))
  bl = rbind(bl, bl_extra)

  # Constraint: root nodes have no incoming edges
  # There is no edge from all nodes to the root nodes
  feat_excl = to_vec(for (x in features) if (! x %in% root_nodes) x)
  for (root in root_nodes) {
    bl_extra = tiers2blacklist(list(root, feat_excl))
    bl = rbind(bl, bl_extra)
  }

  # Constraint: TestLoss has no incoming edge except root nodes
  feat_excl = to_vec(for (x in features) if (x != "TestLoss" && !(startsWith(x, "Train")) && !(startsWith(x, "Test"))
                           && !(x %in% root_nodes)) x)
  bl_extra = tiers2blacklist(list("TestLoss", feat_excl))
  bl = rbind(bl, bl_extra)

  feat_excl = to_vec(for (x in features) if (x != "TestAcc" && !(startsWith(x, "Train")) && !(startsWith(x, "Test"))
                            && !(x %in% root_nodes)) x)
  bl_extra = tiers2blacklist(list("TestAcc", feat_excl))
  bl = rbind(bl, bl_extra)

  # Constraint: There exists no edge from a node that is neither a rootnode nor TestVar to TestVar
  # Because we have computed based on the prediction vectors
  feat_excl = to_vec(for (x in features) if (x != "TestVar" && !(x %in% root_nodes)) x)
  bl_extra = tiers2blacklist(list("TestVar", feat_excl))
  bl = rbind(bl, bl_extra)

  # Constraint: TestBias has no incoming edges except root nodes, TestBias2, TestVar and TestLoss
  feat_excl = to_vec(for (x in features) if (x != "TestBias2" && x != "TestVar" && x != "TestLoss" &&
                                            !(x %in% root_nodes)) x)
  bl_extra = tiers2blacklist(list("TestBias2", feat_excl))
  bl = rbind(bl, bl_extra)


  # Constraint: No edge from TestVar to TestLoss
  bl_extra = tiers2blacklist(list("TestLoss", "TestVar"))
  bl = rbind(bl, bl_extra)


  # Constraint: No edge from TestVar to TestLoss
  # This is a repeat of the constraint of no edge from Test variables to TestLoss
  # bl_extra = tiers2blacklist(list("TestVar", "TestLoss"))
  # bl = rbind(bl, bl_extra)

  # Constraint: No edge from TrainVar to TrainLoss
  bl_extra = tiers2blacklist(list("TrainLoss", "TrainVar"))
  bl = rbind(bl, bl_extra)

  print(sprintf("============== Blacklisted edges ================="))
  print(bl)

  print(sprintf("============== Whitelisted edges ================="))
  print(wl)

  return(list(first=wl, second=bl))
}


run_example_paper = function(dataset) {
  loss = "ce"
  attack = "Oak17Acc"
  cate_results = data.frame(matrix(ncol=5, nrow=0))
  colnames(cate_results) = c('Attack', 'Feature', 'Low Lvl', 'High Lvl', 'ATE')

  features = c("AccDiff", "NumParams")
  att_dataset = dataset[dataset$Loss == loss, c(features, attack)]

  edges = matrix(c("AccDiff", "Oak17Acc",
                   "NumParams", "AccDiff",
                   "NumParams", "AccDiff"),
                  ncol=2, byrow=TRUE, dimnames = list(NULL, c("from", "to")))
  node_list = c(features, attack)
  e = empty.graph(node_list)
  arcs(e) = edges

  fitted = bn.fit(e, att_dataset)
  # Estimate the conditional average
  cate = estimate_effect(fitted, features, dataset, attack)
  for (i in seq_along(cate$X1)) {
    cate_results[nrow(cate_results)+1,] = c(cate$X1[[i]], cate$X2[[i]], cate$X3[[i]], cate$X4[[i]], cate$X5[[i]])
  }

  return(cate_results)
}

check_hypotheses = function(dataset, loss_types, attacks, outdir, discretize_opt, weight_decay) {
  print(names(dataset))
  hypotheses = vector(mode="list", length=9)
  names(hypotheses) = c("Oak17Acc", "Top3MLLeakAcc", "MLLeakAcc", "MLLeakLAcc", "Top3MLLeakLAcc", "ThresholdAcc",
                        "MemguardMemInfAcc", "MemguardNonmemInfAcc", "MemguardInfAcc")
  hypotheses["Oak17Acc"] = list(matrix(c("AccDiff", "Oak17Acc",
                                         "TrainVar", "Oak17Acc",
                                         "TestVar", "Oak17Acc",
                                         "NumParams", "Oak17Acc"),
                                ncol=2, byrow=TRUE, dimnames=list(NULL, c("from", "to"))))
  hypotheses["Top3MLLeakAcc"] = list(matrix(c("AccDiff", "Top3MLLeakAcc",
                                              "NumParams", "Top3MLLeakAcc",
                                              "CentroidDistance.sorted_3.", "Top3MLLeakAcc"),
                                ncol=2, byrow=TRUE, dimnames=list(NULL, c("from", "to"))))
  hypotheses["Top3MLLeakLAcc"] = list(matrix(c("AccDiff", "Top3MLLeakLAcc",
                                              "NumParams", "Top3MLLeakLAcc",
                                              "CentroidDistance.sorted_3.", "Top3MLLeakLAcc"),
                                ncol=2, byrow=TRUE, dimnames=list(NULL, c("from", "to"))))
  hypotheses["MemguardInfAcc"] = list(matrix(c("AccDiff", "MemguardInfAcc",
                                               "NumParams", "MemguardInfAcc",
                                               "CentroidDistance.sorted_3.", "MemguardInfAcc"),
                                    ncol=2, byrow=TRUE, dimnames=list(NULL, c("from", "to"))))
  hypotheses["MemguardMemInfAcc"] = list(matrix(c("AccDiff", "MemguardMemInfAcc",
                                                  "NumParams", "MemguardMemInfAcc",
                                                  "CentroidDistance.sorted_3.", "MemguardMemInfAcc"),
                                    ncol=2, byrow=TRUE, dimnames=list(NULL, c("from", "to"))))
  hypotheses["MemguardNonmemInfAcc"] = list(matrix(c("AccDiff", "MemguardNonmemInfAcc",
                                                     "NumParams", "MemguardNonmemInfAcc",
                                                     "CentroidDistance.sorted_3.", "MemguardNonmemInfAcc"),
                                    ncol=2, byrow=TRUE, dimnames=list(NULL, c("from", "to"))))
  hypotheses["MLLeakAcc"] = list(matrix(c("AccDiff", "MLLeakAcc",
                                          "NumParams", "MLLeakAcc",
                                          "CentroidDistance.origin.", "MLLeakAcc"),
                              ncol=2, byrow=TRUE, dimnames=list(NULL, c("from", "to"))))
  hypotheses["MLLeakLAcc"] = list(matrix(c("AccDiff", "MLLeakLAcc",
                                           "NumParams", "MLLeakLAcc",
                                           "CentroidDistance.origin.", "MLLeakLAcc"),
                              ncol=2, byrow=TRUE, dimnames=list(NULL, c("from", "to"))))
  hypotheses["ThresholdAcc"] = list(matrix(c("LossDiff", "ThresholdAcc"),
                               ncol=2, byrow=TRUE, dimnames=list(NULL, c("from", "to"))))
  corr_results = data.frame(matrix(ncol = 5, nrow = 0))
  colnames(corr_results) = c('Loss', 'Attack', 'Variable', 'Correlation', 'p-value')

  cv_metrics = get_cv_loss_score(discretize_opt)
  loss1 = cv_metrics[[1]]
  loss2 = cv_metrics[[2]]

  for (loss in loss_types) {
    prefix = sprintf("%s/%s", outdir, loss)
    for (attack in attacks) {
      print(attack)
      for (i in 1:nrow(hypotheses[[attack]])) {
        print("========================== Correlation tests ==========================")
        if (loss == "none") {
          att_dataset = dataset[,c(attack, hypotheses[[attack]][i,1])]
        } else {
          att_dataset = dataset[dataset$Loss == loss, c(attack, hypotheses[[attack]][i,1])]
        }
        stat_corr = ci.test(x=attack, y=hypotheses[[attack]][i,1], data=att_dataset, debug=TRUE)
        corr_results[nrow(corr_results)+1,] = c(loss, attack, hypotheses[[attack]][i,1],
                                                stat_corr$statistic, stat_corr$p.value)
      }

    }
  }
  return(corr_results)
}

best_hold_out = function(dataset, alg, prefix, features, attack, bl, wl, thresh, discretize_opt, with_dk) {
  # Depending on the type of network, use different losses
  cv_metrics = get_cv_loss_score(discretize_opt)
  loss1 = cv_metrics[[1]]
  loss2 = cv_metrics[[2]]

  evals = c()
  # bn is fitted (and possibly learned) on the remaining m -nrow(data)
  # the loss function is computed on the m observations in the subsample
  # The overall loss estimate is the average of the k loss estimates from the
  # subsamples.
  # lower log loss values are better
  k = 3
  runs = 20
  test_size = floor(nrow(dataset) * 0.2)
  predcor = structure(numeric(1), names=c(attack))
  mse = structure(numeric(1), names=c(attack))
  # The base loss functions (cor, mse, pred) predict the values of each node
  # just from their parents, which is not meaningful when working on nodes
  # with few or no parents.
  print(test_size)
  print(loss1)
  print(attack)
  xval = bn.cv(data=dataset, bn=alg, algorithm.args=list(blacklist=bl, whitelist=wl),
               loss=loss1, loss.args=list(target=attack),
               fit="mle", method="hold-out", k=k, m=test_size, runs=runs)
  print(xval)

  # Get the average network structure from the cross-validation
  print(sprintf("%s-%s Averaging every possible arcs from the networks from cross-validation", prefix, attack))
  strength <- custom.strength(xval, names(dataset))
  print(sprintf("%s-%s Averaged network threshold: %f", prefix, attack, thresh))
  cv_avg_net = averaged.network(strength)
  file_path = sprintf("%s-%s-cv_avg_net_discovery.pdf", prefix, attack)
  pdf(file=file_path)
  graphviz.plot(cv_avg_net, shape="ellipse", render=TRUE)
  dev.off()
  file_path = sprintf("%s-%s-cv_avg_net_discovery.dot", prefix, attack)
  write.dot(file_path, cv_avg_net)

  predcor[attack] = mean(sapply(xval, function(x) attr(x, "mean")))

  if (!directed(cv_avg_net)) {
    print("There are undirected edges. Skipping this analysis...")
    mse[attack] = -1.0
  } else {
    if (!is.na(loss2)) {
      xval_mse = bn.cv(data=dataset, bn=cv_avg_net,
                      loss=loss2, loss.args=list(target=attack),
                      method="hold-out", k=k, m=test_size, runs=runs)
      mse[attack] = mean(sapply(xval_mse, function(x) attr(x, "mean")))
    }
  }

  return(list(evals=evals, networks=xval, predcor=predcor, mse=mse, cv_avg_net=cv_avg_net))
}


generate_graphs = function(dataset, loss_types, attacks, outdir, alg, discretize_opt, with_dk, weight_decay) {
  results = data.frame(matrix(ncol = 5, nrow = 0))
  cate_results = data.frame(matrix(ncol=6, nrow=0))
  colnames(cate_results) = c('Loss','Attack', 'Feature', 'Low Level', 'High Level', 'CATE')
  colnames(results) = c('DK?', 'Loss', 'Attack', 'Predcor', 'MSE')
  for (loss in loss_types) {
    prefix = sprintf("%s/%s", outdir, loss)
    for (attack in attacks) {
      features = get_features(attack, weight_decay)
      # ================================================================================
      # ========== Proposing a graph that fits the data without U1 and U2 ==============
      # ================================================================================
      if (loss == "none") {
        att_dataset = dataset[,c(features, attack)]
      } else {
        att_dataset = dataset[dataset$Loss == loss, c(features, attack)]
      }

      if (with_dk == TRUE) {
        wl_bl_list = create_wl_bl(features, attack)
        wl = wl_bl_list$first
        bl = wl_bl_list$second
      } else {
        wl = NULL
        bl = NULL
      }

      thresh = NULL

      cv_results = best_hold_out(att_dataset, alg, prefix, features, attack, bl, wl, thresh, discretize_opt, with_dk)
      print(sprintf("============== [DONE with %s attack (bn.cv)] ==================", attack))

      # Check if the resulting graph is directed
      # W/o domain knowledge or extra experiments, we may obtain an undirect graph,
      # hence we cannot perform the next analysis
      if (!directed(cv_results$cv_avg_net)) {
        results[nrow(results)+1,] = c(with_dk, loss, attack, -1, -1)
        print("There are undirected edges. Skipping this analysis...")
        next
      }

      print(cv_results$predcor)
      results[nrow(results)+1,] = c(with_dk, loss, attack, cv_results$predcor[attack],
                                    cv_results$mse[attack])
    }
  }

  return(list(predres=results))
}

main = function(args) {
  if (length(args) < 5) {
    stop("Expecting: Rscript refute_hypotheses.R <path_to_dataset> <outdir_path> <none/loss> <hybrid/discretize/cont> <float_wd>")
  }

  dataset = na.omit(read.csv(args[1]))
  if (args[2] == "run_example") {
    cate_example = run_example_paper(dataset)
    print(cate_example)
    quit()
  }

  outdir = args[2]
  split_on = args[3]
  # This option
  discretize_opt = args[4]
  weight_decay = as.numeric(args[5])

  potential_attacks = c("MLLeakAcc", "MLLeakLAcc",
                        "Top3MLLeakLAcc", "Top3MLLeakAcc",
                        "Oak17Acc",
                        "ThresholdAcc",
                        "MemguardMemInfAcc", "MemguardNonmemInfAcc", "MemguardInfAcc",
                        "RegAttack")
  discrete_cols = c("TrainSize", "EpochNum", "WeightDecay")
  binary_cols = c("Loss", "Scheduler.")

  attacks = list()
  for (attack in potential_attacks) {
    if (attack %in% colnames(dataset)) {
      attacks = c(attacks, attack)
    }
  }
  # For now, we allow to parse the attacks that exist
  print("Found data for attacks: ")
  print(attacks)

  for (discrete_col in discrete_cols) {
    if (discrete_col %in% colnames(dataset) == FALSE) {
      print(sprintf("Column %s not found! Abort.", discrete_col))
    }
  }
  # Make NumParams a continuous variable
  dataset[,"NumParams"] = dataset[,"NumParams"] / max(dataset$NumParams)

  # Filter out the columns that we are aggregating over
  # For learning rate we only have one value for each setup
  # Dataset & Arch => we are trying to understand trends across dataset and arch
  # We only use NumParams as a measure of the model complexity so we leave out arch and width
  ignore_cols = c("lr", "Dataset", "Arch", "Width")
  dataset = dataset[,!names(dataset) %in% ignore_cols]

  if (weight_decay > 0)
    dataset = dataset[dataset$WeightDecay == weight_decay,]

  # Because some of these attacks have only been studied for a particular loss
  # we offer the option to split our data based on loss
  if (split_on == "none")
    loss_types = c("none")
  else
    loss_types = c("ce", "mse")

  # We currently implement these as options
  if (discretize_opt == "discretize") {
    alg = "hc"
    # We are going to try to discretize the data
    cont_data = discretize(dataset[, !names(dataset) %in% c(discrete_cols, binary_cols)], debug=TRUE)
    print(cont_data)
    dataset[, !names(dataset) %in% c(discrete_cols, binary_cols)] = cont_data
    # Make sure that the discrete columns are first interpreted as discrete
    # Depending on whether we try to construct a fully discrete network or not,
    # we have to then process these separately
    dataset[,discrete_cols] = lapply(dataset[,discrete_cols], as.factor)
  } else if (discretize_opt == "hybrid") {
    alg = "mmhc"
    # Change the discrete cols to numeric
    dataset[,discrete_cols] = lapply(dataset[,discrete_cols], as.factor)
  } else if (discretize_opt == "cont") {
    # Change the discrete cols to numeric
    dataset[,discrete_cols] = lapply(dataset[,discrete_cols], as.numeric)
    alg = "hc"
  } else {
    quit("Invalid option for discretize_opt. Use discretize/hybrid/cont")
  }

  corr_results = check_hypotheses(dataset, loss_types, attacks, outdir, discretize_opt, weight_decay)
  print("=========================== FINAL DONE with Correlation Tests =================================")
  print(corr_results)
  print("=========================== FINAL DONE with Correlation Tests =================================")
  filename = sprintf("./corr_results-%s.csv", discretize_opt)
  write.csv(corr_results, filename)

  dk_outdir = sprintf("%s-w_dk-%s-wd_%f", outdir, discretize_opt, weight_decay)
  dir.create(dk_outdir)
  results_w_dk = generate_graphs(dataset, loss_types, attacks, dk_outdir, alg, discretize_opt, TRUE, weight_decay)
  print("=========================== FINAL DONE with DK =================================")
  print(results_w_dk)

  results_filename = sprintf("./results-w_dk-%s-wd_%f.csv", discretize_opt, weight_decay)
  write.csv(results_w_dk$predres, results_filename)

  # We use only the prediction results from here on
  results_w_dk = results_w_dk$predres

  concat_results = data.frame(matrix(ncol=6, nrow=0))
  colnames(concat_results) = c('Loss', 'Attack', 'Max Corr Var', 'Max Corr', 'PredCor-w_dk', 'MSE-w_dk')
  for (attack in attacks) {
    for (loss in loss_types) {
      by_loss = corr_results[corr_results$Loss==loss,]
      by_attack = by_loss[by_loss$Attack==attack,]
      max_corr_row = by_attack[by_attack$Correlation==max(by_attack$Correlation),]
      by_loss = results_w_dk[results_w_dk$Loss == loss,]
      attack_w_dk_pred = by_loss[by_loss$Attack == attack,]['Predcor']
      attack_w_dk_mse = by_loss[by_loss$Attack == attack,]['MSE']
      concat_results[nrow(concat_results)+1,] = c(loss, attack, max_corr_row$Variable,
                                                  max_corr_row$Correlation,
                                                  attack_w_dk_pred,
                                                  attack_w_dk_mse)
    }
  }
  results_filename = sprintf("./results-%s-wd_%f.csv", discretize_opt, weight_decay)
  write.csv(concat_results, results_filename)
}

args = commandArgs(trailingOnly=TRUE)
main(args)
