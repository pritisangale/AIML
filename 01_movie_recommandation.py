
print("Movie Recommendation Assistant")

genre = input("Enter your favorite genre (action/comedy/horror/romance): ").lower()

if genre == "action":
    print("Recommended Movies:")
    print("1. Avengers: Endgame")
    print("2. John Wick")
    print("3. Mad Max: Fury Road")

elif genre == "comedy":
    print("Recommended Movies:")
    print("1. 3 Idiots")
    print("2. The Mask")
    print("3. Jumanji")

elif genre == "horror":
    print("Recommended Movies:")
    print("1. The Conjuring")
    print("2. Insidious")
    print("3. IT")

elif genre == "romance":
    print("Recommended Movies:")
    print("1. Titanic")
    print("2. The Notebook")
    print("3. La La Land")

else:
    print("Sorry! Genre not found.")
    print("Try: action, comedy, horror, or romance.")



#1. print() displays the title Movie Recommendation Assistant.
#2. input() asks the user to enter their favourite movie genre.
#3. .lower() converts the entered genre into lowercase letters.
#4. if genre == "action": checks whether the user selected action. If yes, it displays three action movies.
#5. elif checks other conditions:
#- Comedy → Displays comedy movies.
#- Horror → Displays horror movies.
#- Romance → Displays romantic movies.
#6. else executes when the user enters any genre other than the four given options. It displays "Genre not found."
