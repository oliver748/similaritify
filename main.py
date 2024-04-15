from similaritify import similaritify
import argparse 

def main():
    parser = argparse.ArgumentParser(description='Run image similarity methods.')
    parser.add_argument('-i1', '-img1', type=str, required=True, help='Path to the first image')
    parser.add_argument('-i2', '-img2', type=str, required=True, help='Path to the second image')
    parser.add_argument('-m', '--methods', type=str, required=True, help='Comma-separated list of methods to use, e.g., "akaze,ssim"')
    parser.add_argument('-t', '--target-width', type=int, required=False, default=1280, help='Target width for image processing (1280 by fault)')
    parser.add_argument('-r', '--need-resizing', type=bool, required=False, default=True, help='Resize images (True by default)')

    args = parser.parse_args()

    methods_list = args.methods.split(',')

    s = similaritify.Similaritify(target_width=args.target_width)
    dic = s.run(image_1=args.i1, image_2=args.i2, methods=methods_list, need_resizing=args.need_resizing)


    if dic == {}:
        return None
    
    # print the scores from the methods
    for method, score in dic.items():
        if score is None:
            print(f"{method}: N/A")
            continue

        print(f"{method}: {score:.4f}")


if __name__ == '__main__':
    main()
