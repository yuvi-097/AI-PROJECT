def analyze_csv(filename):
    total_images = 0
    images_with_vehicles = 0
    total_vehicles = 0
    
    with open(filename, 'r') as f:
        # Skip header
        next(f)
        
        # Process each line
        for line in f:
            total_images += 1
            _, boxes = line.strip().split(',')
            boxes = int(boxes)
            total_vehicles += boxes
            if boxes > 0:
                images_with_vehicles += 1
    
    # Calculate statistics
    avg_vehicles = total_vehicles / total_images if total_images > 0 else 0
    
    # Print results
    print(f"Total images: {total_images}")
    print(f"Images with vehicles: {images_with_vehicles}")
    print(f"Total vehicles: {total_vehicles}")
    print(f"Average vehicles per image: {avg_vehicles:.2f}")

if __name__ == "__main__":
    analyze_csv('ground_truth_analysis.csv') 