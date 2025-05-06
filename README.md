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