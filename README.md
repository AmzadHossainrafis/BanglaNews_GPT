## bangla LLM Collection

### looking for active contributors 



1. current available models 
   - Bangla GPT-2 (small, medium, large)(number of heads are 12, 24, 36 respectively)

2. future models 
   - Bangla LLaMA 2(7B, 13B, 65B)
   - Bangla LLaMA 3(7B, 13B, 65B)
   - Bangla qwen 3 (7B, 14B, 70B)

3. dataset
   - demo dataset available inside the dataset folder 
   
   
## installation

1. create an conda environment 
```bash
conda create --name myenv python=3.9 -y 
conda activate myenv
```
2. clone the repo 
```bash 
git clone <repo_link>
cd <repo_name> 

```
3. change the logging dir in `src/Bangala_LLM/utils/logger.py` file
```python
dir = r"/path/to/your/logs/directory"
```


4. install the package
```bash
pip install -e . 

``` 
5. change the config file `configs/config.yaml` as per your requirement


6. install the requirements
```bash
pip install -r requirements.txt
```

7. run the training 
```bash
python src/Bangala_LLM/train.py 
```
