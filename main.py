import argparse
import run_convlstm
import run_convlstm2
import run_lstm

def main():
    parser = argparse.ArgumentParser(description='Run information')
    parser.add_argument('--model', dest='model', type=str, help='Name of the model (convlstm or lstm)')  
    parser.add_argument('--clip',dest='clip',type=float,help='value to clip the data (if no clipping give 1e9)')
    parser.add_argument('--epochs',dest='epochs',type=int,help='number of epochs to run')
    parser.add_argument('--horizon',dest='horizon',type=int,help='horizon of data to predict next value')
    args = parser.parse_args()
    if(args.model=="convlstm"):
        run_convlstm.run(args.clip,7,args.epoch,args.horizon)
    if(args.model=='lstm'):
        run_lstm.run(args.clip,args.epoch,args.horizon)
    if(args.model=='convlstm2'):
        run_convlstm2.run(args.epochs)

if __name__ == "__main__":
    main()



# python3 ./main.py --model convlstm2