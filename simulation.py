from transformers import pipeline, AutoConfig
import time
import torch
from matplotlib import pyplot as plt

word = "bla "

def simulate_for_different_input_lengths(naive_pipe_line,
                                         flash_attention_2_pipeline,
                                         input_lengths,
                                         generation_length):
    naive_runtimes_by_input_length = []
    flash_attn_runtimes_by_input_length = []

    n_repeat = 1
    print("running for different input lengths")


    for input_length in input_lengths:
        print(f"input length: {input_length}")
        text = word * input_length
        text = "<s>[INST] summarize the following text: [/INST] \n" + text

        start = time.time()
        for i in range(n_repeat):
            naive_pipe_line(text, min_new_tokens=generation_length, max_new_tokens=generation_length)
        end = time.time()
        elapsed = end - start
        per_iter = elapsed / n_repeat
        naive_runtimes_by_input_length.append(per_iter)

        start = time.time()
        for i in range(n_repeat):
            flash_attention_2_pipeline(text, min_new_tokens=generation_length, max_new_tokens=generation_length)
        end = time.time()
        elapsed = end - start
        per_iter = elapsed / n_repeat
        flash_attn_runtimes_by_input_length.append(per_iter)

    plt.figure()
    plt.title(f"running times for different input lengths (generation of {generation_length} new token)\n"
              f"{checkpoint}, dtype={naive_pipe_line.model.config.torch_dtype}")
    plt.plot(input_lengths, naive_runtimes_by_input_length, ".-", label="naive")
    plt.plot(input_lengths, flash_attn_runtimes_by_input_length, ".-", label="flash attn 2")
    plt.xlabel("#words in input")
    plt.ylabel("running time [sec]")
    range_ = max(max(naive_runtimes_by_input_length), max(flash_attn_runtimes_by_input_length))
    y_bottom = -0.1 * range_
    y_top = 1.1 * range_
    plt.ylim((y_bottom, y_top))
    plt.legend()
    plt.savefig(f"gen={generation_length}_by_input_length.png")
    plt.show(block=False)
    plt.pause(5)

def simulate_for_different_generation_length(naive_pipe_line,
                                             flash_attention_2_pipeline,
                                             input_length,
                                             generation_lengths):
    naive_runtimes_by_generation_length = []
    flash_attn_runtimes_by_generation_length = []
    text = word * input_length
    text = "<s>[INST] summarize the following text: [/INST] \n" + text
    n_repeat = 1
    print("running for different generation lengths")
    for generation_length in generation_lengths:
        print(f"generation length: {generation_length}")

        start = time.time()
        for i in range(n_repeat):
            naive_pipe_line(text, min_new_tokens=generation_length, max_new_tokens=generation_length)
        end = time.time()
        elapsed = end - start
        per_iter = elapsed / n_repeat
        naive_runtimes_by_generation_length.append(per_iter)

        start = time.time()
        for i in range(n_repeat):
            flash_attention_2_pipeline(text, min_new_tokens=generation_length, max_new_tokens=generation_length)
        end = time.time()
        elapsed = end - start
        per_iter = elapsed / n_repeat
        flash_attn_runtimes_by_generation_length.append(per_iter)

    plt.figure()
    plt.title(f"running times different generation length (for input length of {input_length} words)\n"
              f"{checkpoint}, dtype={naive_pipe_line.model.config.torch_dtype}")
    plt.plot(generation_lengths, naive_runtimes_by_generation_length, ".-", label="naive")
    plt.plot(generation_lengths, flash_attn_runtimes_by_generation_length, ".-", label="flash attn 2")
    plt.xlabel("#newly generated tokens")
    plt.ylabel("running time [sec]")
    range_ = max(max(naive_runtimes_by_generation_length), max(flash_attn_runtimes_by_generation_length))
    y_bottom = -0.1 * range_
    y_top = 1.1 * range_
    plt.ylim((y_bottom, y_top))
    plt.legend()
    plt.savefig(f"input_length={input_length}_by_generation_length.png")
    plt.show(block=False)
    plt.pause(5)


checkpoint = "mistralai/Mistral-7B-Instruct-v0.1"



config = AutoConfig.from_pretrained(checkpoint)
original_dtype = config.torch_dtype

print(f"config dtype: {original_dtype}")
if original_dtype in [torch.float16, torch.bfloat16]:
    print("keeping this dtype")
    torch_dtype = original_dtype
else:
    print("changing to float16")
    torch_dtype = torch.float16
print(f"using dtype: {torch_dtype} (for naive and flash attention alike)", original_dtype)


naive_pipe_line = pipeline("summarization", model=checkpoint, device=0,
                           model_kwargs={"use_flash_attention_2": False,
                                         "torch_dtype": torch_dtype})

flash_attention_2_pipeline = pipeline("summarization", model=checkpoint, device=0,
                                      model_kwargs={"use_flash_attention_2": True,
                                                    "torch_dtype": torch_dtype})

# just run once to get it ready and avoid overheads in first run
text = word * 5
naive_pipe_line(text, min_new_tokens=1, max_new_tokens=1)
flash_attention_2_pipeline(text, min_new_tokens=1, max_new_tokens=1)

#input_lengths = [1] + list(range(100, 1000+1, 100))
#generation_lengths = [1] + list(range(20, 200+1, 20))

input_lengths = [1] + list(range(10, 100, 10)) + list(range(100, 500+1, 100))
generation_lengths = [1, 5, 10] + list(range(20, 100+1, 20))
          
simulate_for_different_generation_length(naive_pipe_line, flash_attention_2_pipeline,
                                         input_length=1, generation_lengths=generation_lengths)

simulate_for_different_generation_length(naive_pipe_line, flash_attention_2_pipeline,
                                         input_length=input_lengths[-1], generation_lengths=generation_lengths)

simulate_for_different_input_lengths(naive_pipe_line, flash_attention_2_pipeline,
                                     input_lengths=input_lengths, generation_length=1)

simulate_for_different_input_lengths(naive_pipe_line, flash_attention_2_pipeline,
                                     input_lengths=input_lengths, generation_length=50)

"""
simulate_for_different_input_lengths(naive_pipe_line, flash_attention_2_pipeline,
                                     input_lengths=input_lengths, generation_length=generation_lengths[-1])
"""












