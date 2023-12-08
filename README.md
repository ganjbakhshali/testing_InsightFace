## bash
run server 
```
uvicorn server:app --reload
```

then run client:
```
python client.py
```
## Test results (on 6000 data)

| Data set | buffalo_l | buffalo_s |
| ---------| ---------| ---------|
| LFW| 83%| 74% |
| CALFW | 80% | 72% |
| CPLFW | 81% | 71% |
