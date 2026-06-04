
#   Indian Name Generator using Neural Networks (PyTorch)
#   Based on next-character prediction with MLP + Embeddings




import torch
import torch.nn as nn
import torch.nn.functional as F
import random



print("Loading Indian names dataset...")

names = [
        "aarav", "aditya", "akash", "amit", "ananya", "anjali", "arjun", "aryan",
        "deepak", "divya", "gaurav", "ishaan", "ishita", "karan", "kavya", "kunal",
        "manish", "meera", "mohit", "neha", "nikhil", "nikita", "pooja", "priya",
        "rahul", "raj", "rajesh", "rakesh", "ravi", "riya", "rohit", "rohan",
        "sachin", "sahil", "sanjay", "sara", "shivam", "shreya", "siddharth", "sneha",
        "sunil", "suresh", "swati", "tanvi", "tarun", "usha", "varun", "vibha",
        "vijay", "vikas", "vishal", "yash", "zara", "abid", "abhishek", "aishwarya",
        "alok", "amisha", "amitabh", "anand", "ankit", "ankita", "anupam", "anurag",
        "archana", "ashish", "ashok", "astha", "avni", "ayesha", "bhavesh", "bhumi",
        "chetan", "chhavi", "disha", "ekta", "farhan", "garima", "girish", "harish",
        "hema", "hemant", "hrithik", "isha", "jagdish", "janhvi", "jasmine", "jayesh",
        "jyoti", "kamal", "kamla", "kapil", "kartik", "kishore", "komal", "krishna",
        "lalit", "lata", "madhur", "mahesh", "malini", "manasi", "manav", "manju",
        "manoj", "meghna", "mihir", "milan", "mohan", "mukesh", "nalini", "namita",
        "nandini", "naresh", "naveen", "nidhi", "nirmal", "nisha", "nitesh", "ojas",
        "pankaj", "paras", "parth", "pavan", "pavani", "pinki", "piyush", "pradeep",
        "pramod", "pranav", "prasad", "pratik", "pratima", "preeti", "pushpa", "radha",
        "raghav", "rajani", "rajiv", "ramesh", "rashmi", "ratan", "rekha", "renuka",
        "ritesh", "ritu", "roopa", "rupal", "rupali", "samir", "sandesh", "sandhya",
        "sandip", "sanjana", "sanjeev", "sanket", "santosh", "sapna", "sarita", "satish",
        "savita", "shailesh", "sharad", "shikha", "shilpa", "shobha", "shruti", "shubham",
        "simran", "smita", "sohan", "sonali", "sonam", "subhash", "sudha", "sumit",
        "sunita", "supriya", "surbhi", "surendra", "sushma", "swapna", "tejas", "tushar",
        "uday", "umesh", "urmila", "varsha", "vedant", "veena", "vicky", "vidya",
        "vikram", "vimal", "vinay", "vineet", "vipul", "vivek", "yamini", "yogesh",
        "abha", "abhay", "aditi", "aditya", "akhil", "akshay", "amita", "amol",
        "amrita", "aniket", "anirudh", "anita", "anju", "anshul", "anuja", "aparna",
        "aradhana", "arun", "asha", "asim", "atharv", "atul", "avani", "ayush",
        "babita", "bhaskar", "bhavana", "bhavesh", "bhoomi", "bhushan", "bipasha",
        "chandni", "chandra", "charu", "chinmay", "chirag", "daksh", "darshan",
        "deepa", "deepti", "devika", "devyani", "dhruv", "dinesh", "dipti",
        "esha", "farah", "farida", "gagan", "gauri", "geeta", "gita", "govind",
        "hardik", "harsha", "heena", "hemangini", "hemanth", "hitesh", "hrishikesh",
        "indira", "indra", "isha", "ishan", "jagdip", "janki", "jasleen", "jaspal",
        "jatin", "jaya", "jayanti", "jaydeep", "jayna", "jignesh", "jitin"
    ]
print(f"Loaded {len(names)} Indian names.")

# Keep only names with regular letters
names = [n for n in names if n.isalpha()]
print(f"Total clean names: {len(names)}")
print("Sample names:", names[:10])


PAD = '.'

# All characters in our vocabulary
all_chars = [PAD] + list("abcdefghijklmnopqrstuvwxyz")
VOCAB_SIZE = len(all_chars)  # = 27

# Map each character to an integer (token ID)
char_to_idx = {ch: i for i, ch in enumerate(all_chars)}
idx_to_char = {i: ch for i, ch in enumerate(all_chars)}

print(f"\nVocabulary size: {VOCAB_SIZE}")
print(f"char_to_idx sample: {dict(list(char_to_idx.items())[:5])}")



CONTEXT_SIZE   = 3    # How many previous characters to look at (block size k)
EMBED_DIM      = 4    # Size of each character's embedding vector (d)
HIDDEN_SIZE    = 64   # Number of neurons in the hidden layer of the MLP
LEARNING_RATE  = 0.01
EPOCHS         = 15000  # Number of training steps
BATCH_SIZE     = 64   # Mini-batch size



def build_dataset(names_list):
    X = [] 
    Y = [] 

    for name in names_list:
       
        context = [char_to_idx[PAD]] * CONTEXT_SIZE

   
        for ch in name + PAD:
            idx = char_to_idx[ch]
            X.append(context)           
            Y.append(idx)              
            context = context[1:] + [idx]  

    X = torch.tensor(X)  
    Y = torch.tensor(Y)   
    return X, Y


random.shuffle(names)
split = int(0.9 * len(names))
train_names = names[:split]
val_names   = names[split:]

X_train, Y_train = build_dataset(train_names)
X_val,   Y_val   = build_dataset(val_names)

print(f"\nTraining samples:   {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Input shape (X):    {X_train.shape}")  # e.g. (44000, 3)
print(f"Output shape (Y):   {Y_train.shape}")  # e.g. (44000,)


class NameMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)

   
        input_size = CONTEXT_SIZE * EMBED_DIM   
        self.fc1 = nn.Linear(input_size, HIDDEN_SIZE)   
        self.fc2 = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE)    

    def forward(self, x):
     
        emb = self.embedding(x)
      

        
        emb = emb.view(emb.shape[0], -1)
       

        
        h = torch.tanh(self.fc1(emb))
   

        logits = self.fc2(h)
       

        return logits


model = NameMLP()


total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel created! Total trainable parameters: {total_params}")




import os

MODEL_FILE = "name_model.pt"   
if os.path.exists(MODEL_FILE):
    
    model.load_state_dict(torch.load(MODEL_FILE))
    print(f"\nLoaded saved model from '{MODEL_FILE}' — skipping training!")

else:
   
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- Starting Training ---")
    print(f"{'Step':>6}  {'Train Loss':>12}  {'Val Loss':>10}")
    print("-" * 35)

    train_losses = []
    val_losses   = []

    for step in range(EPOCHS):

   
        idx = torch.randint(0, X_train.shape[0], (BATCH_SIZE,))
        xb = X_train[idx]
        yb = Y_train[idx]

  
        logits = model(xb)

      
        loss = F.cross_entropy(logits, yb)

      
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

       
        if step % 1000 == 0 or step == EPOCHS - 1:
            train_losses.append(loss.item())
            with torch.no_grad():
                val_logits = model(X_val)
                val_loss   = F.cross_entropy(val_logits, Y_val)
                val_losses.append(val_loss.item())
            print(f"{step:>6}  {loss.item():>12.4f}  {val_loss.item():>10.4f}")

    print("\n--- Training Complete! ---")
    print(f"Final Training Loss:   {train_losses[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")

    # ---- Save the trained model to disk ----
    torch.save(model.state_dict(), MODEL_FILE)
    print(f"\nModel saved to '{MODEL_FILE}' — next run will skip training.")


#  Generate New Names (Autoregressive Sampling)

def generate_name(model, max_len=20):
    model.eval()  


    context = [char_to_idx[PAD]] * CONTEXT_SIZE

    name = ""
    with torch.no_grad():
        while True:
   
            x = torch.tensor([context])  

 
            logits = model(x) 

            probs = F.softmax(logits, dim=1)  

   
            next_idx = torch.multinomial(probs, num_samples=1).item()

           
            if next_idx == char_to_idx[PAD]:
                break

       
            name += idx_to_char[next_idx]

            # Slide the context window forward
            context = context[1:] + [next_idx]

        
            if len(name) >= max_len:
                break

    return name


print("\n--- Generated Names ---")
generated = []
for _ in range(20):
    name = generate_name(model)
    generated.append(name)

for i, name in enumerate(generated, 1):
    print(f"  {i:>2}. {name.capitalize()}")


#  Plot the Loss Curve + Embedding Visualization


import matplotlib.pyplot as plt

# Training & Validation Loss Curve


if 'train_losses' in dir() or 'train_losses' in locals() or 'train_losses' in globals():

    # X axis: the step numbers where we recorded loss (every 1000 steps)
    steps_recorded = list(range(0, EPOCHS, 1000)) + [EPOCHS - 1]
    steps_recorded = steps_recorded[:len(train_losses)]  # make sure lengths match

    plt.figure(figsize=(9, 5))

    # Plot training loss as a blue line
    plt.plot(steps_recorded, train_losses,
             color='steelblue', linewidth=2, marker='o', markersize=4,
             label='Training Loss')

    # Plot validation loss as an orange dashed line
    plt.plot(steps_recorded, val_losses,
             color='darkorange', linewidth=2, marker='o', markersize=4,
             linestyle='--', label='Validation Loss')

    plt.title("Training & Validation Loss Over Time", fontsize=14, fontweight='bold')
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.show()
    print("\nLoss curve saved as 'loss_curve.png'")

else:
    print("\n(Loss curve only available right after training — delete name_model.pt and rerun to see it)")


#  Learned Character Embeddings

emb_weights = model.embedding.weight.detach().numpy()


if EMBED_DIM > 2:
    
    emb_tensor = model.embedding.weight.detach()
    emb_centered = emb_tensor - emb_tensor.mean(0)        
    _, _, V = torch.pca_lowrank(emb_centered, q=2)        
    emb_2d = (emb_centered @ V).numpy()                     
    x_coords = emb_2d[:, 0]
    y_coords = emb_2d[:, 1]
    axis_label = "PCA Dimension"
else:
    # Already 2D, plot directly
    x_coords = emb_weights[:, 0]
    y_coords = emb_weights[:, 1]
    axis_label = "Embedding Dimension"


vowels = set('aeiou')
dot    = {'.'}

fig, ax = plt.subplots(figsize=(10, 7))

for i, ch in enumerate(all_chars):
   
    if ch in vowels:
        color = 'crimson'       
        size  = 120
    elif ch in dot:
        color = 'black'         
        size  = 140
    else:
        color = 'steelblue'     
        size  = 100

    ax.scatter(x_coords[i], y_coords[i], color=color, s=size, zorder=3)


    ax.annotate(
        ch,
        (x_coords[i], y_coords[i]),
        textcoords="offset points",
        xytext=(6, 6),
        fontsize=13,
        fontweight='bold',
        color=color
    )

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson',
           markersize=10, label='Vowels (a, e, i, o, u)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue',
           markersize=10, label='Consonants'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
           markersize=10, label='. (end-of-name token)'),
]
ax.legend(handles=legend_elements, fontsize=11, loc='upper right')

ax.set_title("Learned Character Embeddings\n(red = vowels, blue = consonants)",
             fontsize=14, fontweight='bold')
ax.set_xlabel(f"{axis_label} 1", fontsize=12)
ax.set_ylabel(f"{axis_label} 2", fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("embeddings.png", dpi=150)
plt.show()
print("Embedding plot saved as 'embeddings.png'")
print("\nLook at the plot — vowels (red) should be clustered together,")
print("consonants (blue) in a separate group, and '.' off on its own.")


#  Interactive Mode — Generate names on demand

print("\n--- Interactive Name Generator ---")
print("Press Enter to generate a name, or type 'q' to quit.\n")

while True:
    user_input = input("Generate a name? (Enter / q to quit): ").strip()
    if user_input.lower() == 'q':
        break
    name = generate_name(model)
    print(f"  → {name.capitalize()}\n")

print("Done! Thanks for using the Indian Name Generator.")