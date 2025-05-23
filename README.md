# README

## Set Up
```
python3.11 -m venv venv
. venv/bin/activate
pip3.11 install -e .
```


## 1. Get started
### 1.1 run baseline agent: the prompt agent
```
python run_demo_baseline.py \
    --browser-mode chromium \
    --storage-state shopping.json \
    --starting-url "http://xwebarena.pathonai.org:7770/grocery-gourmet-food/food-beverage-gifts/herb-spice-seasoning-gifts.html?product_list_mode=list" \
    --agent-type "PromptAgent" \
    --action_generation_model "gpt-4o-mini" \
    --goal "Can you add the item Yemeni Hawayij on this page, to my cart?" \
    --evaluator-type "BaseEvaluator" \
    --eval-url "http://xwebarena.pathonai.org:7770/checkout/cart" \
    --eval-criteria "check if Yemeni Hawayij is in the cart"
```

### 1.2 run tree search
#### 1.2.1 BFS/ DFS
```
python run_demo_treesearch.py \
    --browser-mode chromium \
    --storage-state shopping.json \
    --starting-url "http://xwebarena.pathonai.org:7770/grocery-gourmet-food/food-beverage-gifts/herb-spice-seasoning-gifts.html?product_list_mode=list" \
    --agent-type "SimpleSearchAgent" \
    --action_generation_model "gpt-4o-mini" \
    --goal "Can you add the item Yemeni Hawayij on this page, to my cart?" \
    --iterations 3 \
    --max_depth 3 \
    --search_algorithm bfs
```

#### 1.2.2 MCTS

```
python run_demo_treesearch.py \
    --browser-mode chromium \
    --storage-state shopping.json \
    --starting-url "http://xwebarena.pathonai.org:7770/" \
    --agent-type "MCTSAgent" \
    --action_generation_model "gpt-4o-mini" \
    --goal "search running shoes, click on the first result" \
    --iterations 3 \
    --max_depth 3
```
#### 1.2.3 older version of LATS/ MCTS agents in the visual tree search demo
```
python run_demo_treesearch.py \
    --browser-mode chromium \
    --storage-state shopping.json \
    --starting-url "http://xwebarena.pathonai.org:7770/" \
    --agent-type "LATSAgent" \
    --action_generation_model "gpt-4o-mini" \
    --goal "search running shoes, click on the first result" \
    --iterations 3 \
    --max_depth 3
```

```
python run_demo_treesearch.py \
    --browser-mode chromium \
    --storage-state shopping.json \
    --starting-url "http://xwebarena.pathonai.org:7770/" \
    --agent-type "RMCTSAgent" \
    --action_generation_model "gpt-4o-mini" \
    --goal "search running shoes, click on the first result" \
    --iterations 3 \
    --max_depth 3
```

## 2. Run Evaluation on the xwebarena benchmark
```
python run_xwebarena_eval.py --browser-mode chromium --agent-type "PromptAgent" --action_generation_model "gpt-4o-mini" --config-file ./xwebarena_evaluation_suite/configs/wa/test_webarena/124.json
```


## 3. Run Evaluation on the webshop benchmark
### 3.1 Version 1 webshop
```
# Prompt Agent on WebShop
# run single task
python run_webshop_eval.py --starting-url "http://128.105.144.173:3000/fixed_0"  

# run batch of tasks
python run_webshop_eval.py --starting-url "http://128.105.144.173:3000/fixed_0"  --batch-start 1 --batch-end 5

# TreeSearch Agent on WebShop
# run single task
python run_webshop_eval_tree_search.py --search-algorithm bfs --browser-mode chromium/browserbase

# run batch of tasks
python run_webshop_eval_tree_search.py --search-algorithm dfs --batch-start 0 --batch-end 5
```

### 3.2 Version 2 webshop
```
python run_webshop_eval_new.py --starting-url "http://54.224.220.64:3000/fixed_0"  
```


```
python run_webshop_eval_tree_search_new.py --starting-url "http://54.224.220.64:3000/fixed_0"  
```

run batch of tasks the same as run_webshop_eval.py
