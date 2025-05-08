# README

## Get started
```
python run_demo_treesearch_async.py \
    --browser-mode chromium \
    --storage-state shopping.json \
    --starting-url "http://xwebarena.pathonai.org:7770/" \
    --agent-type "SimpleSearchAgent" \
    --action_generation_model "gpt-4o-mini" \
    --goal "search running shoes, click on the first result" \
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
    --eval-url "http://128.105.146.96:7770/checkout/cart" \
    --eval-criteria "check if Yemeni Hawayij is in the cart"
```