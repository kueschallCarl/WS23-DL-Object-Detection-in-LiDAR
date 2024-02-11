import cProfile
import pstats
from start_run_inference import main as start_inference

def main(): 
    cProfile.run('start_inference()', 'inference_profile.stats')

    p = pstats.Stats('inference_profile.stats')
    p.sort_stats('cumtime').print_stats(30)  # Print the top 10 time-consuming functions

if __name__ == '__main__':  # Corrected line
    main()
