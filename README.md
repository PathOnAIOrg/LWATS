# README

## Set Up
```
python3.11 -m venv venv
. venv/bin/activate
pip3.11 install -e .
```


## Get started
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

```
# run single task
python run_webshop_eval.py --starting-url "http://54.224.220.64:3000/fixed_0"  --agent-type PromptAgent

# run batch of tasks
python run_webshop_eval.py --starting-url "http://54.224.220.64:3000/fixed_0" --agent-type PromptAgent --batch-start 1 --batch-end 5
```