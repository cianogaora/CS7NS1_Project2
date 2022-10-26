# CS7NS1_Project2

## Instructions

First, generate a training set using the following command:

`./generate.py --width 128 --height 64 --length 6 --symbols symbols.txt --count <number of desired images> --output-dir output_folder`

Use this command again to generate a validation set

If the script is not executable,first make it executable using the command:

`chmod +x generate.py` 


Next, train the model using the following command:

`./train.py --width 128 --height 64 --length 6 --symbols symbols.txt --batch-size 32 --epochs 5 -- output-model model.h5 --train-dataset /path/to/training/data --validate-dataset path/to/validation/data`

To classify, run the command: 
`./classify.py --model-name model --captcha-dir path/to/test/images/ --output output.txt --symbols symbols.txt`

Predictions will be outputted into the file 'output.txt', which can be converted to a csv and uploaded to Submitty

