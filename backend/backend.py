import re, json, uuid, asyncio, logging, hashlib, io, random
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx, uvicorn
import base64

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("rag-v7")
import os
from dotenv import load_dotenv
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MDL   = "llama3-70b-8192"
CHROMA_DIR = "./chroma_db"
EMBED_MDL  = "all-MiniLM-L6-v2"
COLLECTION = "recipes_rag_v7"

app = FastAPI(title="Recipe RAG v7")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── INGREDIENT ONTOLOGY ──────────────────────────────────────────────────────
ING_MAP = {
    "chicken":["chicken","boneless chicken","chicken breast","chicken thigh","chicken pieces","grilled chicken","chicken strips","shredded chicken"],
    "beef":["beef","ground beef","minced beef","steak","beef mince","beef strips"],
    "pork":["pork","pork belly","bacon","pancetta","ham","prosciutto"],
    "lamb":["lamb","ground lamb","lamb chop","mutton","goat meat"],
    "salmon":["salmon","salmon fillet","smoked salmon","atlantic salmon"],
    "tuna":["tuna","canned tuna","tuna steak"],
    "cod":["cod","cod fillet"],
    "fish":["fish","fish fillet","sea bass","halibut","tilapia","basa","snapper","white fish"],
    "shrimp":["shrimp","prawn","prawns","tiger prawn","king prawn","jumbo shrimp","scampi"],
    "crab":["crab","crab meat"],
    "squid":["squid","calamari"],
    "eggs":["eggs","egg","egg yolk","egg white","boiled egg"],
    "milk":["milk","whole milk","skim milk","dairy milk"],
    "butter":["butter","unsalted butter","salted butter"],
    "cream":["cream","heavy cream","whipping cream","fresh cream","single cream","double cream"],
    "yogurt":["yogurt","curd","greek yogurt","plain yogurt","dahi","sour cream","hung curd"],
    "cheese":["cheese","cheddar","mozzarella","feta","gouda","brie"],
    "parmesan":["parmesan","parmigiano","grana padano","pecorino","parmesan cheese"],
    "paneer":["paneer","cottage cheese","fresh paneer"],
    "ghee":["ghee","clarified butter","desi ghee"],
    "rice":["rice","basmati rice","cooked rice","jasmine rice","arborio rice","white rice","brown rice","leftover rice"],
    "pasta":["pasta","spaghetti","penne","fettuccine","linguine","fusilli","noodles","macaroni","tagliatelle"],
    "bread":["bread","sandwich bread","sourdough","baguette","whole wheat bread","toast","brioche"],
    "flour":["flour","all purpose flour","plain flour","wheat flour","maida","bread flour"],
    "oats":["oats","rolled oats","oatmeal","porridge oats","quick oats"],
    "pizza dough":["pizza dough","pizza base"],
    "tomato":["tomato","tomatoes","cherry tomatoes","canned tomatoes","tomato puree","tomato paste","crushed tomatoes","tinned tomatoes"],
    "onion":["onion","onions","red onion","white onion","yellow onion","spring onion","shallot","scallion","chives"],
    "garlic":["garlic","garlic cloves","minced garlic","garlic paste","garlic powder","roasted garlic"],
    "ginger":["ginger","fresh ginger","ginger paste","ground ginger","ginger powder"],
    "spinach":["spinach","baby spinach","fresh spinach","frozen spinach","palak"],
    "carrot":["carrot","carrots","baby carrots","grated carrot","shredded carrot"],
    "potato":["potato","potatoes","sweet potato","aloo","yam","baby potatoes"],
    "mushroom":["mushrooms","mushroom","portobello","button mushrooms","cremini","shiitake","oyster mushroom"],
    "bell pepper":["bell pepper","capsicum","red pepper","green pepper","yellow pepper"],
    "zucchini":["zucchini","courgette"],
    "avocado":["avocado","avocados","ripe avocado","hass avocado"],
    "cucumber":["cucumber","english cucumber"],
    "corn":["corn","sweet corn","corn kernels","maize"],
    "peas":["peas","green peas","frozen peas","sugar snap peas"],
    "celery":["celery","celery stalks"],
    "broccoli":["broccoli","broccoli florets"],
    "cauliflower":["cauliflower","cauliflower florets","gobi","phool gobi"],
    "eggplant":["eggplant","aubergine","brinjal","baingan"],
    "cabbage":["cabbage","red cabbage","napa cabbage","shredded cabbage"],
    "apple":["apple","apples","green apple","red apple","granny smith","apple slices","fuji apple"],
    "mango":["mango","mangoes","mango pulp","alphonso","ripe mango","frozen mango"],
    "banana":["banana","bananas","ripe banana","plantain","overripe banana"],
    "strawberry":["strawberry","strawberries","fresh strawberries","frozen strawberries","sliced strawberries"],
    "lemon":["lemon","lemon juice","lemon zest","citrus"],
    "lime":["lime","lime juice","lime zest"],
    "orange":["orange","orange juice","orange zest","mandarin"],
    "blueberry":["blueberry","blueberries","frozen blueberries"],
    "raspberry":["raspberry","raspberries","fresh raspberries"],
    "peach":["peach","peaches","canned peach"],
    "oil":["oil","vegetable oil","cooking oil","sunflower oil","canola oil","neutral oil"],
    "olive oil":["olive oil","extra virgin olive oil","evoo"],
    "coconut milk":["coconut milk","coconut cream","full fat coconut milk"],
    "soy sauce":["soy sauce","soya sauce","light soy sauce","dark soy sauce","tamari"],
    "sugar":["sugar","granulated sugar","white sugar","brown sugar","caster sugar","powdered sugar","icing sugar"],
    "honey":["honey","raw honey","organic honey","maple syrup"],
    "salt":["salt","sea salt","kosher salt","table salt","himalayan salt"],
    "black pepper":["black pepper","pepper","ground pepper","cracked pepper","white pepper"],
    "cumin":["cumin","cumin seeds","jeera","ground cumin"],
    "cinnamon":["cinnamon","ground cinnamon","cinnamon stick","cinnamon powder"],
    "turmeric":["turmeric","haldi","ground turmeric","turmeric powder"],
    "garam masala":["garam masala","garam masala powder","mixed spice"],
    "chili":["chili","chilli","red chili","green chili","chili powder","chilli flakes","cayenne","kashmiri chili","red pepper flakes"],
    "paprika":["paprika","smoked paprika","sweet paprika"],
    "coriander powder":["coriander powder","dhania powder","ground coriander"],
    "cardamom":["cardamom","green cardamom","cardamom powder","elaichi"],
    "saffron":["saffron","kesar","saffron strands"],
    "bay leaf":["bay leaf","bay leaves","tej patta"],
    "oregano":["oregano","dried oregano"],
    "thyme":["thyme","fresh thyme","dried thyme"],
    "rosemary":["rosemary","fresh rosemary"],
    "dill":["dill","fresh dill","dill weed"],
    "mint":["mint","fresh mint","mint leaves","pudina"],
    "basil":["basil","fresh basil","thai basil","basil leaves"],
    "cilantro":["cilantro","coriander","fresh coriander","coriander leaves","dhania"],
    "baking powder":["baking powder","raising agent"],
    "baking soda":["baking soda","bicarbonate of soda","bicarb"],
    "vanilla extract":["vanilla extract","vanilla essence","vanilla","vanilla bean"],
    "cocoa powder":["cocoa powder","unsweetened cocoa","cacao powder"],
    "chocolate":["chocolate","dark chocolate","milk chocolate","chocolate chips","semi-sweet chocolate"],
    "cream cheese":["cream cheese","philadelphia","mascarpone","neufchatel"],
    "tomato sauce":["tomato sauce","marinara","passata","pizza sauce"],
    "caramel sauce":["caramel sauce","caramel","dulce de leche","toffee sauce"],
    "whipped cream":["whipped cream","cool whip"],
    "green curry paste":["green curry paste","thai green curry paste"],
    "fish sauce":["fish sauce","nam pla","thai fish sauce"],
    "hot sauce":["hot sauce","sriracha","tabasco"],
    "nuts":["nuts","walnuts","almonds","cashews","peanuts","pine nuts","pistachios","mixed nuts","pecans"],
    "sesame oil":["sesame oil","toasted sesame oil","asian sesame oil"],
    "sesame":["sesame","sesame seeds","tahini"],
    "chickpeas":["chickpeas","chana","garbanzo beans","canned chickpeas","dried chickpeas"],
    "lentils":["lentils","red lentil","green lentil","black lentil","masoor dal","moong dal","dal"],
    "kidney beans":["kidney beans","rajma","canned kidney beans","red kidney beans"],
    "black beans":["black beans","canned black beans"],
    "tofu":["tofu","firm tofu","silken tofu","extra firm tofu"],
    "vanilla ice cream":["vanilla ice cream","ice cream","vanilla icecream","gelato"],
    "wafers":["vanilla wafers","wafers","graham crackers","digestive biscuits","biscuits"],
    "white wine":["white wine","dry white wine","cooking wine"],
    "vegetable stock":["vegetable stock","vegetable broth","veg stock","veggie broth"],
    "chicken broth":["chicken broth","chicken stock","chicken bouillon"],
    "biryani masala":["biryani masala","biryani spice mix","biryani spice"],
    "chana masala":["chana masala","chole masala"],
    "spring onion":["spring onion","scallion","green onion"],
    "taco shells":["taco shells","tortillas","corn tortillas","flour tortillas"],
}
SEAFOOD = {"fish","salmon","tuna","cod","shrimp","crab","squid","seafood","halibut","tilapia","sea bass"}

# Build reverse canonical map
CANONICAL: dict[str, str] = {}
for canon, variants in ING_MAP.items():
    CANONICAL[canon.lower().strip()] = canon
    for v in variants:
        CANONICAL[v.lower().strip()] = canon

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())

def canonicalize(raw: str) -> str:
    r = norm(raw)
    if r in CANONICAL: return CANONICAL[r]
    for alias in sorted(CANONICAL, key=len, reverse=True):
        if len(alias) > 2 and alias in r: return CANONICAL[alias]
    return r

# ─── RECIPE DATASET ───────────────────────────────────────────────────────────
RECIPES: list[dict] = [
    {"id":"r01","name":"Chicken Biryani","cuisine":"Indian","diet":["non-veg"],"time":"60 min","servings":4,"tags":["rice","main","dinner","spicy","chicken"],"ingredients":{"chicken":"500g","basmati rice":"2 cups","onion":"3","tomato":"2","yogurt":"1 cup","garlic":"6 cloves","ginger":"2 inch","biryani masala":"3 tbsp","ghee":"3 tbsp","mint":"handful","saffron":"pinch"},"steps":["Marinate chicken with yogurt, spices, ginger-garlic 2 hours.","Parboil basmati rice 70% with whole spices.","Fry onions in ghee until deep golden, reserve half.","Cook chicken with tomatoes on medium heat until oil separates.","Layer rice over chicken, top with saffron milk and fried onions. Seal and dum cook 25 min on low flame."]},
    {"id":"r02","name":"Butter Chicken","cuisine":"Indian","diet":["non-veg"],"time":"45 min","servings":4,"tags":["curry","main","dinner","mild","chicken"],"ingredients":{"chicken":"600g","butter":"4 tbsp","cream":"½ cup","tomato puree":"1 cup","onion":"2","garlic":"6 cloves","ginger":"1 inch","garam masala":"1 tsp","chili":"2 tsp"},"steps":["Marinate and grill chicken until charred.","Sauté onion-garlic-ginger in butter; add tomato puree, simmer 15 min.","Blend sauce completely smooth.","Add grilled chicken and cream; simmer 10 min.","Finish with butter and fenugreek leaves. Serve with naan."]},
    {"id":"r03","name":"Thai Green Curry","cuisine":"Thai","diet":["non-veg"],"time":"30 min","servings":4,"tags":["curry","main","dinner","chicken","coconut","quick"],"ingredients":{"chicken":"400g","coconut milk":"400ml","green curry paste":"3 tbsp","bell pepper":"1","fish sauce":"2 tbsp","basil":"handful","zucchini":"1","oil":"1 tbsp"},"steps":["Fry green curry paste in oil 1 min until fragrant.","Add half coconut milk; cook until oil rises.","Add chicken; cook through 8 min.","Add remaining coconut milk, vegetables, fish sauce.","Finish with Thai basil. Serve with jasmine rice."]},
    {"id":"r04","name":"Chicken Caesar Salad","cuisine":"American","diet":["non-veg"],"time":"25 min","servings":2,"tags":["salad","lunch","quick","chicken","healthy"],"ingredients":{"chicken":"2 breasts","romaine lettuce":"1 head","parmesan":"50g","bread":"1 cup croutons","yogurt":"4 tbsp caesar dressing","lemon":"1","olive oil":"2 tbsp"},"steps":["Grill chicken 6 min per side; rest 5 min, slice.","Toss romaine with caesar dressing.","Top with chicken, croutons, parmesan.","Finish with lemon squeeze and cracked black pepper."]},
    {"id":"r05","name":"Chicken Soup","cuisine":"American","diet":["non-veg"],"time":"50 min","servings":6,"tags":["soup","lunch","dinner","healthy","chicken","comfort"],"ingredients":{"chicken":"500g","carrot":"3","celery":"3 stalks","onion":"1","garlic":"4 cloves","chicken broth":"6 cups","pasta":"200g egg noodles","thyme":"4 sprigs"},"steps":["Simmer chicken in broth with thyme 30 min.","Remove and shred chicken; discard bones.","Sauté onion, celery, carrot until soft.","Return broth, chicken; add noodles.","Cook 8 min. Season generously and serve."]},
    {"id":"r06","name":"Chicken Fried Rice","cuisine":"Chinese","diet":["non-veg"],"time":"20 min","servings":3,"tags":["rice","main","dinner","lunch","chicken","quick"],"ingredients":{"chicken":"300g","rice":"3 cups cooked","eggs":"2","soy sauce":"3 tbsp","garlic":"3 cloves","spring onion":"4","sesame oil":"1 tsp","oil":"2 tbsp"},"steps":["Cook diced chicken in smoking wok until golden.","Scramble eggs; push to side.","Add garlic and cold rice; stir-fry 3 min breaking clumps.","Season with soy sauce; toss well.","Finish with sesame oil and spring onion."]},
    {"id":"r07","name":"Grilled Salmon","cuisine":"Mediterranean","diet":["non-veg","pescatarian"],"time":"20 min","servings":4,"tags":["seafood","dinner","quick","healthy","fish","salmon"],"ingredients":{"salmon":"4 fillets","butter":"3 tbsp","lemon":"2","garlic":"3 cloves","dill":"handful","olive oil":"2 tbsp"},"steps":["Pat salmon dry; season with salt and pepper.","Grill 4 min per side on oiled hot pan.","Melt butter in same pan; add garlic 1 min.","Add lemon juice and fresh dill.","Spoon butter sauce generously over salmon."]},
    {"id":"r08","name":"Prawn Stir-Fry","cuisine":"Chinese","diet":["non-veg","pescatarian"],"time":"15 min","servings":3,"tags":["seafood","dinner","quick","prawn","shrimp"],"ingredients":{"shrimp":"400g prawns","bell pepper":"2","garlic":"4 cloves","ginger":"1 inch","soy sauce":"3 tbsp","sesame oil":"1 tsp","spring onion":"4","oil":"2 tbsp","chili":"1 tsp"},"steps":["Heat wok to smoking. Add oil, garlic, ginger 30 sec.","Add prawns; cook 2 min until pink and curled.","Add sliced bell peppers; stir-fry 2 min keeping crisp.","Add soy sauce and chili flakes; toss well.","Finish with sesame oil and spring onion."]},
    {"id":"r09","name":"Garlic Butter Shrimp Pasta","cuisine":"Italian","diet":["non-veg","pescatarian"],"time":"25 min","servings":4,"tags":["pasta","seafood","dinner","quick","shrimp","prawn"],"ingredients":{"shrimp":"400g","pasta":"400g","butter":"4 tbsp","garlic":"6 cloves","lemon":"1","parmesan":"60g","olive oil":"2 tbsp","chili":"pinch","basil":"handful"},"steps":["Boil pasta al dente; save 1 cup pasta water.","Cook garlic in butter+olive oil 1 min.","Add shrimp; cook 2 min per side until pink.","Add lemon juice and chili flakes.","Toss with pasta, parmesan, pasta water. Top with basil."]},
    {"id":"r10","name":"Fish Tacos","cuisine":"Mexican","diet":["non-veg","pescatarian"],"time":"25 min","servings":4,"tags":["seafood","lunch","quick","fish","tacos"],"ingredients":{"fish":"500g white fish","taco shells":"8","yogurt":"½ cup","lime":"2","cabbage":"1 cup shredded","garlic":"2 cloves","cumin":"1 tsp","chili":"1 tsp","olive oil":"2 tbsp","cilantro":"handful"},"steps":["Season fish; pan-fry in olive oil 3 min per side.","Mix yogurt with lime juice and garlic for crema.","Warm taco shells.","Fill with fish and shredded cabbage.","Drizzle crema; top with cilantro and lime."]},
    {"id":"r11","name":"Tuna Pasta Salad","cuisine":"Western","diet":["non-veg","pescatarian"],"time":"20 min","servings":4,"tags":["salad","pasta","seafood","lunch","quick","tuna"],"ingredients":{"tuna":"2 cans","pasta":"300g","celery":"3 stalks","onion":"½","lemon":"1","yogurt":"4 tbsp mayonnaise","salt":"to taste","black pepper":"to taste"},"steps":["Cook pasta al dente; cool under cold water.","Flake tuna; dice celery and red onion finely.","Mix mayo with lemon juice for dressing.","Combine pasta, tuna, vegetables.","Fold in dressing; chill 30 min before serving."]},
    {"id":"r12","name":"Prawn Curry","cuisine":"Indian","diet":["non-veg","pescatarian"],"time":"35 min","servings":4,"tags":["curry","seafood","dinner","spicy","prawn","shrimp"],"ingredients":{"shrimp":"500g prawns","coconut milk":"400ml","onion":"2","tomato":"3","garlic":"5 cloves","ginger":"1 inch","turmeric":"1 tsp","chili":"2 tsp","cumin":"1 tsp","coriander powder":"2 tsp","oil":"3 tbsp"},"steps":["Sauté onion golden; add ginger-garlic paste.","Add tomatoes and all spices; cook until oil separates.","Pour coconut milk; simmer 5 min.","Add prawns; cook exactly 4-5 min until pink.","Garnish with cilantro. Serve with rice."]},
    {"id":"r13","name":"Baked Cod with Herbs","cuisine":"Mediterranean","diet":["non-veg","pescatarian"],"time":"30 min","servings":4,"tags":["seafood","dinner","healthy","fish","cod","baked"],"ingredients":{"cod":"4 fillets","olive oil":"3 tbsp","garlic":"4 cloves","lemon":"1","thyme":"4 sprigs","rosemary":"2 sprigs","tomato":"2","salt":"to taste","black pepper":"to taste"},"steps":["Preheat oven 200°C. Pat cod fillets dry.","Place in baking dish; drizzle olive oil.","Top with sliced garlic, herbs, and tomatoes.","Squeeze lemon; season with salt and pepper.","Bake 15-18 min until fish flakes easily."]},
    {"id":"r14","name":"Salmon Fried Rice","cuisine":"Asian","diet":["non-veg","pescatarian"],"time":"20 min","servings":3,"tags":["rice","seafood","dinner","quick","salmon","fish"],"ingredients":{"salmon":"300g","rice":"3 cups cooked","eggs":"2","soy sauce":"3 tbsp","garlic":"3 cloves","ginger":"1 inch","sesame oil":"1 tsp","spring onion":"4","oil":"2 tbsp"},"steps":["Cook salmon; flake into large chunks.","Scramble eggs in hot wok; push aside.","Add garlic, ginger; then cold rice; stir-fry 3 min.","Add soy sauce; toss vigorously.","Fold in salmon gently. Finish with sesame oil."]},
    {"id":"r15","name":"Palak Paneer","cuisine":"Indian","diet":["vegetarian"],"time":"30 min","servings":3,"tags":["curry","vegetarian","dinner","main","paneer","quick"],"ingredients":{"paneer":"250g","spinach":"500g","onion":"2","tomato":"1","garlic":"4 cloves","ginger":"1 inch","cream":"2 tbsp","cumin":"1 tsp","garam masala":"½ tsp","butter":"2 tbsp"},"steps":["Blanch spinach 2 min; shock in ice water; blend smooth.","Sauté cumin in butter; add onion until golden.","Add ginger-garlic, tomato; cook down 5 min.","Pour spinach puree; simmer 5 min.","Add paneer cubes, cream, garam masala."]},
    {"id":"r16","name":"Paneer Tikka Masala","cuisine":"Indian","diet":["vegetarian"],"time":"50 min","servings":4,"tags":["curry","vegetarian","dinner","main","paneer"],"ingredients":{"paneer":"300g","yogurt":"½ cup","bell pepper":"2","onion":"2","tomato":"3","cream":"¼ cup","garam masala":"1.5 tsp","chili":"1.5 tsp","garlic":"5 cloves","butter":"2 tbsp"},"steps":["Marinate paneer in yogurt and spices 30 min.","Grill paneer and peppers until charred.","Build masala: onion, garlic, tomatoes, spices.","Blend sauce smooth; add cream and paneer.","Simmer 10 min. Serve with garlic naan."]},
    {"id":"r17","name":"Chana Masala","cuisine":"Indian","diet":["vegetarian","vegan"],"time":"40 min","servings":4,"tags":["curry","vegan","dinner","main","chickpeas"],"ingredients":{"chickpeas":"2 cans","onion":"2","tomato":"3","garlic":"5 cloves","ginger":"1 inch","cumin":"1 tsp","coriander powder":"2 tsp","chana masala":"2 tsp","oil":"3 tbsp"},"steps":["Sauté cumin; add onion until deep golden (15 min).","Add ginger-garlic paste; cook 2 min.","Add tomatoes and spices; cook until oil separates.","Add chickpeas with ½ cup water; simmer 20 min.","Mash some chickpeas to thicken; garnish with cilantro."]},
    {"id":"r18","name":"Dal Makhani","cuisine":"Indian","diet":["vegetarian"],"time":"90 min","servings":4,"tags":["lentils","vegetarian","dinner","main","comfort"],"ingredients":{"lentils":"1 cup black","kidney beans":"¼ cup","butter":"3 tbsp","cream":"¼ cup","tomato puree":"1 cup","onion":"2","garlic":"5 cloves","garam masala":"1 tsp"},"steps":["Soak and pressure cook lentils+beans 45 min until very soft.","Sauté onion dark golden; add garlic, spices, tomato puree.","Combine lentils with masala; simmer 20 min stirring.","Add cream and butter; slow cook 10 min.","Serve with butter naan. Better the next day!"]},
    {"id":"r19","name":"Mushroom Risotto","cuisine":"Italian","diet":["vegetarian"],"time":"35 min","servings":4,"tags":["rice","vegetarian","dinner","main","mushroom","comfort"],"ingredients":{"mushroom":"300g","rice":"1.5 cups arborio","onion":"1","garlic":"3 cloves","white wine":"½ cup","parmesan":"80g","butter":"4 tbsp","vegetable stock":"5 cups"},"steps":["Keep stock warm throughout cooking.","Sauté mushrooms golden; set aside.","Cook onion+garlic in butter; add rice; toast 1 min.","Add wine; then stock ladle by ladle, stirring constantly.","After 18 min fold in mushrooms, parmesan, remaining butter."]},
    {"id":"r20","name":"Shakshuka","cuisine":"Middle Eastern","diet":["vegetarian"],"time":"30 min","servings":3,"tags":["breakfast","egg","vegetarian","quick","spicy"],"ingredients":{"eggs":"4","tomato":"400g canned","bell pepper":"1","onion":"1","garlic":"4 cloves","cumin":"1 tsp","paprika":"1 tsp","olive oil":"2 tbsp"},"steps":["Sauté onion and pepper in olive oil 5 min.","Add garlic and spices; cook 1 min until fragrant.","Pour canned tomatoes; simmer 10 min until thick.","Make 4 wells; crack eggs in; cover.","Cook 5-8 min. Serve with crusty bread."]},
    {"id":"r21","name":"Avocado Toast","cuisine":"Western","diet":["vegetarian"],"time":"10 min","servings":2,"tags":["breakfast","quick","healthy","vegetarian","avocado"],"ingredients":{"avocado":"1","bread":"2 slices","eggs":"2","lemon":"½","chili":"pinch"},"steps":["Toast bread until golden and crispy.","Mash avocado with lemon juice, salt, pepper.","Poach eggs in simmering water 3 min.","Spread avocado thick on toast.","Top with egg, chili flakes, and extra lemon."]},
    {"id":"r22","name":"Pesto Pasta","cuisine":"Italian","diet":["vegetarian"],"time":"20 min","servings":4,"tags":["pasta","lunch","dinner","quick","vegetarian"],"ingredients":{"pasta":"400g","basil":"2 cups","nuts":"3 tbsp pine nuts","parmesan":"60g","garlic":"2 cloves","olive oil":"½ cup"},"steps":["Boil pasta al dente; save 1 cup pasta water.","Toast pine nuts until golden; watch carefully.","Blend basil, nuts, garlic, parmesan until smooth.","Drizzle olive oil while blending; season.","Toss pasta with pesto; loosen with pasta water."]},
    {"id":"r23","name":"Margherita Pizza","cuisine":"Italian","diet":["vegetarian"],"time":"25 min","servings":2,"tags":["main","dinner","baked","vegetarian","pizza"],"ingredients":{"pizza dough":"300g","tomato sauce":"½ cup","cheese":"200g mozzarella","basil":"handful","olive oil":"2 tbsp"},"steps":["Preheat oven 250°C with stone inside.","Stretch dough to 30cm — don't use rolling pin.","Spread thin tomato sauce; tear mozzarella over.","Drizzle olive oil generously.","Bake 10-12 min. Top with fresh basil immediately."]},
    {"id":"r24","name":"Vegetable Fried Rice","cuisine":"Chinese","diet":["vegetarian","vegan"],"time":"20 min","servings":3,"tags":["rice","lunch","dinner","vegan","quick","vegetarian"],"ingredients":{"rice":"3 cups cooked","eggs":"3","carrot":"1","peas":"½ cup","spring onion":"4","soy sauce":"3 tbsp","garlic":"3 cloves","sesame oil":"1 tsp"},"steps":["Heat wok to smoking hot.","Scramble eggs; push to edges.","Add garlic, diced carrot, peas; stir-fry 2 min.","Add cold rice; break clumps vigorously.","Season with soy sauce; finish with sesame oil."]},
    {"id":"r25","name":"Egg Fried Rice","cuisine":"Chinese","diet":["vegetarian"],"time":"15 min","servings":2,"tags":["breakfast","rice","quick","egg","vegetarian"],"ingredients":{"rice":"2 cups cooked","eggs":"3","soy sauce":"2 tbsp","garlic":"2 cloves","spring onion":"3","sesame oil":"1 tsp","oil":"1 tbsp"},"steps":["Heat oil in wok on highest heat.","Scramble eggs until just set; push aside.","Add garlic and cold rice; stir-fry 3 min hard.","Add soy sauce; toss and fry 1 more min.","Fold eggs in; finish with sesame oil and spring onion."]},
    {"id":"r26","name":"Masala Omelette","cuisine":"Indian","diet":["vegetarian"],"time":"10 min","servings":1,"tags":["breakfast","quick","egg","vegetarian","indian"],"ingredients":{"eggs":"3","onion":"1 small","tomato":"1 small","chili":"1","cilantro":"handful","turmeric":"pinch","oil":"1 tsp"},"steps":["Beat eggs with salt and turmeric until frothy.","Stir in finely chopped onion, tomato, chili, cilantro.","Heat oil in non-stick pan on medium.","Pour egg mixture; spread evenly.","Cook 2 min; fold in half. Serve hot."]},
    {"id":"r27","name":"Spaghetti Carbonara","cuisine":"Italian","diet":["non-veg"],"time":"25 min","servings":4,"tags":["pasta","dinner","quick","main","pork"],"ingredients":{"pasta":"400g spaghetti","pork":"150g pancetta","eggs":"4","parmesan":"100g","garlic":"2 cloves","black pepper":"2 tsp"},"steps":["Boil spaghetti al dente; save 1 cup pasta water.","Fry pancetta and garlic 8 min until crispy.","Whisk egg yolks with parmesan and black pepper.","OFF HEAT: toss pasta with pancetta then egg mix.","Add pasta water tablespoon by tablespoon for silky sauce."]},
    {"id":"r28","name":"Beef Tacos","cuisine":"Mexican","diet":["non-veg"],"time":"25 min","servings":4,"tags":["main","dinner","lunch","quick","beef","tacos"],"ingredients":{"beef":"500g ground","taco shells":"8","onion":"1","garlic":"3 cloves","cumin":"1 tsp","chili":"1.5 tsp","tomato":"2","cheese":"100g cheddar","yogurt":"4 tbsp sour cream","lime":"1"},"steps":["Brown beef; break into fine pieces; drain fat.","Add onion, garlic, cumin, chili; cook 2 min.","Warm taco shells in oven 3 min until crispy.","Fill with beef, diced tomato, cheddar, sour cream.","Squeeze lime over each. Serve immediately."]},
    {"id":"r29","name":"Tomato Soup","cuisine":"Western","diet":["vegetarian","vegan"],"time":"30 min","servings":4,"tags":["soup","lunch","dinner","vegan","comfort","tomato"],"ingredients":{"tomato":"8 large","onion":"1","garlic":"4 cloves","vegetable stock":"2 cups","olive oil":"2 tbsp","basil":"handful","sugar":"1 tsp"},"steps":["Roast tomatoes and garlic with olive oil at 200°C 20 min.","Sauté onion until soft.","Add roasted tomatoes and stock; simmer 10 min.","Blend completely smooth; strain for silky texture.","Season; add sugar; garnish with basil and cream."]},
    {"id":"r30","name":"Apple Crumble","cuisine":"British","diet":["vegetarian"],"time":"45 min","servings":6,"tags":["dessert","baked","sweet","apple"],"ingredients":{"apple":"4 large","sugar":"4 tbsp","cinnamon":"1 tsp","lemon":"1","flour":"1 cup","butter":"80g cold","oats":"½ cup"},"steps":["Slice apples; toss with sugar, cinnamon, lemon juice.","Spread apple mixture in buttered baking dish.","Rub cold butter into flour, oats, brown sugar until crumbly.","Spread crumble evenly over apples — don't press.","Bake 180°C 30 min until golden and bubbling."]},
    {"id":"r31","name":"Apple Cinnamon Pancakes","cuisine":"American","diet":["vegetarian"],"time":"25 min","servings":4,"tags":["breakfast","sweet","quick","apple","dessert"],"ingredients":{"apple":"2","flour":"1.5 cups","eggs":"2","milk":"1 cup","sugar":"2 tbsp","cinnamon":"1 tsp","baking powder":"2 tsp","butter":"2 tbsp","vanilla extract":"1 tsp"},"steps":["Grate apple; squeeze out excess moisture.","Mix flour, sugar, cinnamon, baking powder.","Whisk eggs, milk, vanilla, melted butter.","Fold wet into dry; add grated apple — don't overmix.","Cook on medium 2-3 min per side. Serve with maple syrup."]},
    {"id":"r32","name":"Strawberry Cheesecake","cuisine":"American","diet":["vegetarian"],"time":"60 min","servings":8,"tags":["dessert","sweet","no-bake","strawberry","cheesecake"],"ingredients":{"strawberry":"300g","cream cheese":"400g","cream":"200ml whipping","wafers":"200g","butter":"80g","sugar":"100g","lemon":"1","vanilla extract":"1 tsp"},"steps":["Crush biscuits; mix with melted butter; press into 20cm tin. Chill 30 min.","Beat cream cheese with sugar, vanilla, lemon until smooth.","Whip cream to soft peaks; fold into cream cheese.","Spread over chilled base; smooth top. Chill 4+ hours.","Arrange halved strawberries decoratively before serving."]},
    {"id":"r33","name":"Strawberry Smoothie","cuisine":"Western","diet":["vegetarian","vegan"],"time":"5 min","servings":2,"tags":["drink","healthy","quick","strawberry","breakfast"],"ingredients":{"strawberry":"2 cups","banana":"1","milk":"1 cup","honey":"1 tbsp","vanilla extract":"½ tsp"},"steps":["Hull strawberries; freeze for 1 hour for thicker texture.","Add strawberries and banana to blender.","Pour in milk, honey, vanilla.","Blend on high 60 sec until smooth.","Taste; adjust sweetness. Serve chilled."]},
    {"id":"r34","name":"Strawberry Shortcake","cuisine":"American","diet":["vegetarian"],"time":"40 min","servings":8,"tags":["dessert","baked","sweet","strawberry"],"ingredients":{"strawberry":"400g","flour":"2 cups","sugar":"7 tbsp","baking powder":"1 tbsp","butter":"6 tbsp cold","cream":"1 cup","vanilla extract":"1 tsp"},"steps":["Macerate strawberries with 3 tbsp sugar 30 min.","Mix flour, 4 tbsp sugar, baking powder; cut in cold butter.","Add cream+vanilla; mix until just combined.","Drop onto baking sheet; bake 200°C 15 min.","Split warm biscuits; fill with strawberries and whipped cream."]},
    {"id":"r35","name":"Carrot Cake","cuisine":"American","diet":["vegetarian"],"time":"60 min","servings":12,"tags":["dessert","baked","sweet","carrot"],"ingredients":{"carrot":"3 cups grated","flour":"2 cups","sugar":"1.5 cups","eggs":"4","oil":"1 cup","cinnamon":"2 tsp","baking soda":"2 tsp","vanilla extract":"1 tsp","cream cheese":"400g","butter":"100g"},"steps":["Preheat 180°C. Whisk eggs, oil, sugar, vanilla.","Mix flour, cinnamon, baking soda; fold into wet.","Fold in grated carrot. Divide between two greased tins.","Bake 30-35 min. Test with skewer — must come out clean.","Beat cream cheese+butter+icing sugar. Frost cooled cake."]},
    {"id":"r36","name":"Carrot Halwa","cuisine":"Indian","diet":["vegetarian"],"time":"45 min","servings":6,"tags":["dessert","sweet","indian","carrot","halwa"],"ingredients":{"carrot":"500g grated","milk":"2 cups","sugar":"½ cup","ghee":"3 tbsp","cardamom":"4 pods","nuts":"handful"},"steps":["Cook grated carrot with milk on medium, stirring every 5 min, until dry (25 min).","Add ghee; cook 5 min — carrot changes colour.","Add sugar; stir and cook 10 min until halwa leaves pan sides.","Add crushed cardamom and chopped nuts.","Serve warm or at room temperature."]},
    {"id":"r37","name":"Banana Pancakes","cuisine":"American","diet":["vegetarian"],"time":"15 min","servings":2,"tags":["breakfast","sweet","quick","banana","dessert"],"ingredients":{"banana":"2 ripe","oats":"1 cup","eggs":"2","milk":"¼ cup","cinnamon":"½ tsp","baking powder":"1 tsp","vanilla extract":"½ tsp"},"steps":["Mash very ripe bananas completely smooth.","Blend oats to fine flour.","Combine banana, eggs, milk, vanilla; fold in oat flour, baking powder, cinnamon.","Cook small pancakes on medium 2 min per side until bubbles form.","Serve with honey or maple syrup."]},
    {"id":"r38","name":"Mango Lassi","cuisine":"Indian","diet":["vegetarian"],"time":"5 min","servings":2,"tags":["drink","sweet","quick","mango","indian"],"ingredients":{"mango":"2 ripe","yogurt":"1 cup","milk":"½ cup","sugar":"2 tbsp","cardamom":"pinch"},"steps":["Add ripe mango chunks and yogurt to blender.","Add milk, sugar, and cardamom.","Blend on high 60 sec until frothy.","Taste; adjust sweetness.","Serve immediately in chilled tall glasses."]},
    {"id":"r39","name":"Apple Smoothie","cuisine":"Western","diet":["vegetarian","vegan"],"time":"5 min","servings":2,"tags":["drink","healthy","quick","apple","breakfast"],"ingredients":{"apple":"2","banana":"1","milk":"1 cup","honey":"1 tbsp","cinnamon":"pinch"},"steps":["Core and chop apples — no need to peel.","Add apple and banana to blender.","Pour in milk, honey, cinnamon.","Blend on high until completely smooth.","Serve immediately or chill. Shake if stored."]},
    {"id":"r40","name":"Apple Ice Cream Sundae","cuisine":"American","diet":["vegetarian"],"time":"10 min","servings":2,"tags":["dessert","sweet","quick","apple"],"ingredients":{"apple":"1","vanilla ice cream":"3 scoops","caramel sauce":"2 tbsp","whipped cream":"¼ cup","cinnamon":"pinch","butter":"1 tbsp"},"steps":["Slice apple; sauté in butter 3-4 min until golden.","Scoop ice cream into serving bowl.","Top with warm caramel apple slices.","Add whipped cream.","Dust with cinnamon and drizzle caramel sauce."]},
    {"id":"r41","name":"Strawberry Ice Cream","cuisine":"American","diet":["vegetarian"],"time":"10 min + freeze","servings":6,"tags":["dessert","sweet","frozen","strawberry"],"ingredients":{"strawberry":"400g","cream":"400ml whipping","milk":"200ml","sugar":"100g","vanilla extract":"1 tsp"},"steps":["Blend strawberries with sugar until smooth puree.","Whip cream to soft peaks.","Fold in strawberry puree, milk, vanilla.","Pour into container; press cling film to surface.","Freeze 6 hours, stirring every 2 hours to prevent ice crystals."]},
    {"id":"r42","name":"Chocolate Cake","cuisine":"American","diet":["vegetarian"],"time":"50 min","servings":12,"tags":["dessert","baked","sweet","chocolate"],"ingredients":{"flour":"2 cups","cocoa powder":"¾ cup","sugar":"2 cups","eggs":"3","milk":"1 cup","oil":"½ cup","baking powder":"2 tsp","baking soda":"1 tsp","vanilla extract":"1 tsp","butter":"100g"},"steps":["Sift flour, cocoa, baking powder, baking soda together.","Whisk eggs, milk, oil, and vanilla.","Fold wet into dry until smooth batter.","Bake in greased tins 180°C 32-35 min.","Cool completely; frost with chocolate buttercream."]},
    {"id":"r43","name":"Classic Omelette","cuisine":"French","diet":["vegetarian"],"time":"5 min","servings":1,"tags":["breakfast","quick","egg","vegetarian","french"],"ingredients":{"eggs":"3","butter":"1 tbsp","salt":"pinch","black pepper":"pinch"},"steps":["Beat eggs with salt and pepper until uniform.","Melt butter in hot pan until foamy — very hot.","Pour eggs; stir gently while shaking pan continuously.","Fold when edges set but center still custardy.","Slide onto warm plate immediately."]},
    {"id":"r44","name":"Banana Oat Smoothie","cuisine":"Western","diet":["vegetarian","vegan"],"time":"5 min","servings":2,"tags":["drink","healthy","breakfast","quick","banana"],"ingredients":{"banana":"2","oats":"½ cup","milk":"1.5 cups","honey":"1 tbsp","cinnamon":"pinch","vanilla extract":"½ tsp"},"steps":["Blend oats with half the milk 30 sec.","Add banana, remaining milk, honey, vanilla, cinnamon.","Blend 60 sec until completely smooth.","Taste; adjust sweetness with honey.","Serve immediately in tall glasses."]},
    {"id":"r45","name":"Lamb Rogan Josh","cuisine":"Indian","diet":["non-veg"],"time":"70 min","servings":4,"tags":["curry","main","dinner","spicy","lamb"],"ingredients":{"lamb":"600g","onion":"3","tomato":"2","yogurt":"1 cup","garlic":"6 cloves","ginger":"2 inch","garam masala":"2 tsp","chili":"2 tsp","oil":"4 tbsp","cardamom":"4 pods"},"steps":["Brown lamb in batches on high heat — don't crowd.","Cook onion until very dark golden in same pot.","Add garlic-ginger, tomatoes, spices, yogurt.","Return browned lamb; add 1 cup water.","Simmer covered 45-50 min until fork-tender."]},
    {"id":"r46","name":"Greek Salad","cuisine":"Mediterranean","diet":["vegetarian"],"time":"10 min","servings":4,"tags":["salad","lunch","quick","healthy","vegetarian","mediterranean"],"ingredients":{"tomato":"4","cucumber":"1","onion":"1 red","cheese":"150g feta","olive oil":"3 tbsp","lemon":"1","oregano":"1 tsp","salt":"to taste"},"steps":["Cut tomatoes into chunky wedges.","Slice cucumber into half-moons; thinly slice red onion.","Combine in large bowl.","Crumble feta in large pieces over top.","Drizzle olive oil, squeeze lemon, season with oregano."]},
    {"id":"r47","name":"Caprese Salad","cuisine":"Italian","diet":["vegetarian"],"time":"10 min","servings":4,"tags":["salad","lunch","quick","vegetarian","italian"],"ingredients":{"tomato":"4 large","cheese":"300g fresh mozzarella","basil":"handful","olive oil":"3 tbsp","salt":"to taste","black pepper":"to taste"},"steps":["Use best quality tomatoes and mozzarella available.","Slice both into ½-inch rounds of similar size.","Alternate on serving platter slightly overlapping.","Tuck fresh basil between each slice.","Drizzle with best olive oil; season with flaky salt."]},
    {"id":"r48","name":"Red Lentil Soup","cuisine":"Middle Eastern","diet":["vegetarian","vegan"],"time":"35 min","servings":4,"tags":["soup","lunch","dinner","vegan","healthy","lentil"],"ingredients":{"lentils":"1.5 cups red","onion":"2","garlic":"4 cloves","tomato":"2","cumin":"1.5 tsp","turmeric":"1 tsp","lemon":"1","olive oil":"3 tbsp","vegetable stock":"6 cups"},"steps":["Rinse red lentils until water runs clear.","Sauté onion golden; add garlic 2 min.","Add cumin, turmeric; cook 1 min.","Add tomatoes, lentils, stock; simmer 20 min until dissolved.","Blend half for texture; add lemon juice. Serve."]},
    {"id":"r49","name":"Vegetable Curry","cuisine":"Indian","diet":["vegetarian","vegan"],"time":"35 min","servings":4,"tags":["curry","dinner","vegan","main","vegetarian"],"ingredients":{"potato":"3","carrot":"2","peas":"1 cup","onion":"2","tomato":"3","garlic":"4 cloves","ginger":"1 inch","cumin":"1 tsp","turmeric":"1 tsp","garam masala":"1 tsp","oil":"3 tbsp","coconut milk":"200ml"},"steps":["Sauté onion; add garlic-ginger paste.","Add spices and tomatoes; cook 10 min until oil separates.","Add potatoes, carrots, 1 cup water; cover; simmer 15 min.","Add peas and coconut milk; simmer 5 min.","Garnish with cilantro. Serve with rice."]},
    {"id":"r50","name":"Stuffed Bell Peppers","cuisine":"Mediterranean","diet":["vegetarian"],"time":"45 min","servings":4,"tags":["main","dinner","baked","vegetarian","bell pepper"],"ingredients":{"bell pepper":"4 large","rice":"1 cup cooked","cheese":"100g feta","tomato":"2","onion":"1","garlic":"3 cloves","olive oil":"2 tbsp","oregano":"1 tsp","basil":"handful"},"steps":["Preheat 180°C. Cut tops off peppers; remove seeds.","Sauté onion and garlic in olive oil until soft.","Mix rice, feta, diced tomatoes, herbs with onion mix.","Fill peppers tightly with mixture.","Bake 30 min until peppers are tender and filling is hot."]},
    {"id":"r51","name":"Cauliflower Soup","cuisine":"Western","diet":["vegetarian","vegan"],"time":"30 min","servings":4,"tags":["soup","healthy","vegetarian","vegan","cauliflower","quick"],"ingredients":{"cauliflower":"1 head","onion":"1","garlic":"4 cloves","vegetable stock":"4 cups","olive oil":"2 tbsp","cream":"4 tbsp","black pepper":"to taste","thyme":"3 sprigs"},"steps":["Sauté onion in olive oil 5 min until soft.","Add garlic and thyme; cook 1 min.","Add cauliflower florets and stock.","Simmer 20 min until cauliflower is very tender.","Blend smooth; stir in cream; season generously."]},
    {"id":"r52","name":"Aloo Gobi","cuisine":"Indian","diet":["vegetarian","vegan"],"time":"30 min","servings":4,"tags":["curry","vegetarian","vegan","cauliflower","potato","quick","indian"],"ingredients":{"cauliflower":"1 head","potato":"3","onion":"1","tomato":"2","garlic":"3 cloves","ginger":"1 inch","turmeric":"1 tsp","cumin":"1 tsp","chili":"1 tsp","coriander powder":"2 tsp","oil":"3 tbsp","cilantro":"handful"},"steps":["Heat oil; add cumin seeds until they splutter.","Add onion until golden; add ginger-garlic paste.","Add tomatoes and all spices; cook 5 min until oil separates.","Add potato and cauliflower; cover and cook 15-18 min.","Garnish with fresh cilantro."]},
    {"id":"r53","name":"Roasted Cauliflower","cuisine":"Mediterranean","diet":["vegetarian","vegan"],"time":"35 min","servings":4,"tags":["side","healthy","vegetarian","vegan","cauliflower"],"ingredients":{"cauliflower":"1 head","olive oil":"3 tbsp","garlic":"4 cloves","lemon":"1","paprika":"1 tsp","cumin":"1 tsp","salt":"to taste","black pepper":"to taste","cilantro":"handful"},"steps":["Preheat oven 220°C — very hot for proper roasting.","Cut cauliflower into even florets.","Toss with olive oil, garlic, paprika, cumin, salt, pepper.","Spread on baking sheet in single layer — don't crowd.","Roast 25-30 min, tossing halfway, until charred and golden."]},
]

# Pre-compute canonical ingredient sets
for r in RECIPES:
    r["_ci"] = {canonicalize(norm(k)) for k in r.get("ingredients", {})}
    r["_sf"] = bool(r["_ci"] & SEAFOOD)

# ─── SUBSTITUTIONS ────────────────────────────────────────────────────────────
SUBS = {
    "paneer":["Firm tofu (1:1, press dry — vegan)","Halloumi (1:1, saltier, grills well)","Chicken breast (1:1, non-veg)"],
    "chicken":["Turkey (1:1, leaner)","Firm tofu (1:1, marinate well, vegan)","Paneer (1:1, vegetarian)","Chickpeas (1 cup per 200g, vegan)"],
    "beef":["Lamb (1:1, richer)","Portobello mushroom (1:1, vegan umami)","Jackfruit young (1:1, vegan)"],
    "salmon":["Tuna steak (1:1)","Cod fillet (1:1, milder)","Firm tofu (1:1, vegan)"],
    "fish":["Tofu (1:1, vegan — pan fry firm)","Jackfruit (1:1)","Hearts of palm (1:1)"],
    "shrimp":["Chicken strips (1:1)","Scallops (1:1)","Firm tofu cubes (1:1, vegan)"],
    "butter":["Ghee (¾:1, higher smoke point)","Coconut oil (¾:1, vegan)","Olive oil (¾:1)"],
    "cream":["Coconut cream (1:1, vegan)","Cashew cream (blend soaked cashews+water)","Greek yogurt (1:1, reduce heat)"],
    "eggs":["Flax egg (1 tbsp ground flax + 3 tbsp water)","Silken tofu (¼ cup = 1 egg)","Banana mashed (¼ cup, baking only)"],
    "milk":["Oat milk (1:1, creamiest)","Almond milk (1:1, lighter)","Soy milk (1:1, highest protein)","Coconut milk (1:1, richer)"],
    "parmesan":["Pecorino Romano (1:1, sharper)","Nutritional yeast (2 tbsp per 30g, vegan)"],
    "apple":["Pear (1:1)","Peach (1:1, softer)","Quince (1:1, tarter)"],
    "mango":["Peach (1:1)","Papaya (1:1)","Apple (1:1, less sweet)"],
    "strawberry":["Raspberries (1:1)","Peach (1:1, sweeter)","Mango (1:1, tropical)"],
    "vanilla ice cream":["Gelato (1:1, denser)","Frozen yogurt (1:1, tangier)","Coconut ice cream (1:1, vegan)"],
    "yogurt":["Sour cream (1:1)","Coconut yogurt (1:1, vegan)","Buttermilk (¾:1)"],
    "pasta":["Zucchini noodles (1:1, low-carb)","Rice noodles (1:1, gluten-free)","Spaghetti squash (roast and scrape)"],
    "rice":["Quinoa (1:1, more protein)","Cauliflower rice (1:1, low-carb)","Couscous (1:1, faster)"],
    "coconut milk":["Cashew cream (1:1)","Heavy cream (1:1, non-vegan)","Oat cream (1:1)"],
    "spinach":["Kale (1:1, cook longer)","Swiss chard (1:1)","Frozen spinach (½ weight, squeeze dry)"],
    "chickpeas":["White beans (1:1)","Lentils (1:1, cook faster)","Edamame (1:1)"],
    "garlic":["Garlic powder (⅛ tsp per clove)","Shallots (1 small per clove)","Asafoetida/hing (tiny pinch)"],
    "lemon":["Lime (1:1)","White wine vinegar (½ amount)","Apple cider vinegar (½ amount)"],
    "carrot":["Parsnip (1:1)","Sweet potato (1:1)","Butternut squash (1:1)"],
    "cream cheese":["Mascarpone (1:1)","Vegan cream cheese (1:1)","Ricotta (1:1, lighter)"],
    "cauliflower":["Broccoli (1:1)","Romanesco (1:1)","Turnip (for soups)"],
    "chocolate":["Carob powder (1:1, caffeine-free)","Dark cocoa + sugar (adjust to taste)"],
    "nuts":["Sunflower seeds (1:1, nut-free)","Pumpkin seeds (1:1)","Toasted coconut flakes (1:1)"],
}

# ─── MEAL TYPE MAP ────────────────────────────────────────────────────────────
MEAL_KW = {
    "dessert":  ["dessert","sweet","sweets","cake","pudding","treat","ice cream","cheesecake","crumble","halwa","cookie","brownie"],
    "breakfast":["breakfast","morning","brunch"],
    "soup":     ["soup"],
    "drink":    ["drink","smoothie","lassi","juice","shake","beverage"],
    "salad":    ["salad"],
    "snack":    ["snack","snacks","appetizer"],
    "lunch":    ["lunch"],
    "dinner":   ["dinner","supper"],
}
QUICK_KW = ["quick","fast","easy","under 30","under 20","under 15","simple","rapid"]

def detect_meal(t: str) -> Optional[str]:
    t = norm(t)
    for meal, kws in MEAL_KW.items():
        if any(k in t for k in kws): return meal
    return None

def detect_diet_filter(t: str) -> Optional[str]:
    t = norm(t)
    if re.search(r'\bvegan\b', t): return "vegan"
    if re.search(r'\bveg(etarian)?\b|\bno meat\b|\bplant.based\b', t): return "vegetarian"
    if re.search(r'\bnon[- ]?veg\b|\bmeat\b', t): return "non-veg"
    if re.search(r'\bpescatarian\b', t): return "pescatarian"
    return None

# ─── SESSIONS ─────────────────────────────────────────────────────────────────
SESSIONS: dict[str, dict] = {}
def get_sess(sid: str) -> dict:
    if sid not in SESSIONS:
        SESSIONS[sid] = {"exclude": [], "diet": [], "history": [], "pending_gen": None}
    return SESSIONS[sid]

# ─── NOISE WORDS ──────────────────────────────────────────────────────────────
NOISE = {"i","have","want","make","cook","need","some","the","and","with","for","a","an","me","please","can","my","is","no","any","all","do","what","give","suggest","show","recipes","recipe","dish","food","using","would","like","also","but","not","only","just","something","cup","cups","tbsp","tsp","grams","kg","ml","few","bit","little","help","hello","hi","hey","ok","yes","yeah","thanks","great","awesome","random","list","dishes","get","find","search","quick","easy","simple","spicy","healthy","veg","non","vegetarian","vegan","dinner","lunch","breakfast","dessert"}
_TOKENS = sorted({norm(v) for vs in ING_MAP.values() for v in vs} | {norm(k) for r in RECIPES for k in r.get("ingredients", {})}, key=len, reverse=True)

def extract_ingredients(text: str) -> list[str]:
    t = norm(text); found = []
    for tok in _TOKENS:
        if re.search(r'(?<![a-z])' + re.escape(tok) + r'(?![a-z])', t):
            found.append(tok)
            t = re.sub(r'(?<![a-z])' + re.escape(tok) + r'(?![a-z])', " " * len(tok), t)
    for part in re.split(r'[,&]|\band\b', re.sub(r'[^a-z,& \-]+', " ", t)):
        p = part.strip()
        if p and len(p) > 2 and p not in NOISE and not p.isdigit():
            found.append(p)
    seen, out = set(), []
    for f in found:
        c = canonicalize(f)
        if c and c not in seen: seen.add(c); out.append(c)
    return out

# ═══════════════════════════════════════════════════════════════════════════════
# TRUE RAG CORE — ChromaDB Embedding Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

_col   = None
_model = None

def recipe_to_document(r: dict) -> str:
    """Convert recipe to rich semantic text for embedding.
    This is the KEY step — good document = good retrieval."""
    ings   = " ".join(r.get("ingredients", {}).keys())
    tags   = " ".join(r.get("tags", []))
    steps  = " ".join(r.get("steps", []))
    diets  = " ".join(r.get("diet", []))
    return (
        f"Recipe Name: {r['name']}. "
        f"Cuisine: {r['cuisine']}. "
        f"Diet: {diets}. "
        f"Category tags: {tags}. "
        f"Ingredients: {ings}. "
        f"Cooking time: {r.get('time','unknown')}. "
        f"Method: {steps}"
    )

def build_chroma() -> None:
    """Build or load ChromaDB collection with recipe embeddings.
    Runs ONCE — subsequent server starts load from disk."""
    global _col, _model
    if _col is not None:
        return

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer

        log.info("🔧 Initialising ChromaDB persistent client…")
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        col    = client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"}   # ← cosine similarity
        )

        # Hash dataset to detect changes → only re-embed if needed
        ds_hash = hashlib.md5(
            json.dumps([recipe_to_document(r) for r in RECIPES]).encode()
        ).hexdigest()

        already_indexed = False
        if col.count() > 0:
            try:
                meta = col.get(ids=["__meta__"], include=["metadatas"])
                already_indexed = (
                    meta["metadatas"] and
                    meta["metadatas"][0].get("hash") == ds_hash
                )
            except Exception:
                pass

        if not already_indexed:
            log.info("📐 Building recipe embeddings (one-time — will be cached)…")
            mdl   = SentenceTransformer(EMBED_MDL)
            _model = mdl

            # Convert each recipe → rich text document
            docs  = [recipe_to_document(r) for r in RECIPES]

            # Encode all recipes at once (batch for speed)
            embeddings = mdl.encode(
                docs,
                normalize_embeddings=True,   # ← unit vectors for cosine similarity
                batch_size=32,
                show_progress_bar=False
            ).tolist()

            # Upsert into ChromaDB
            col.upsert(
                ids        = [r["id"] for r in RECIPES],
                documents  = docs,
                embeddings = embeddings,
                metadatas  = [{"name": r["name"], "cuisine": r["cuisine"]} for r in RECIPES]
            )
            # Store hash marker
            col.upsert(
                ids=["__meta__"],
                documents=["metadata"],
                metadatas=[{"hash": ds_hash}]
            )
            log.info(f"✅ {len(RECIPES)} recipes embedded and persisted to {CHROMA_DIR}")
            log.info("   Next server start will SKIP embedding — loaded from disk.")

        else:
            log.info(f"✅ ChromaDB: loading {col.count()-1} cached embeddings from {CHROMA_DIR}")
            _model = SentenceTransformer(EMBED_MDL)

        _col = col

    except Exception as e:
        log.warning(f"⚠️  ChromaDB/SentenceTransformer unavailable: {e}")
        log.warning("    Falling back to keyword-only mode.")

def semantic_search(query: str, top_k: int = 15) -> list[tuple[str, float]]:
    """
    Core RAG retrieval:
      1. Embed the user query with the same model
      2. ChromaDB finds top-K recipes by cosine similarity
      3. Returns list of (recipe_id, similarity_score)
    """
    if _col is None or _model is None:
        return []
    try:
        # Embed query
        q_embedding = _model.encode(
            [query],
            normalize_embeddings=True
        ).tolist()

        # Query ChromaDB — returns results sorted by cosine distance
        results = _col.query(
            query_embeddings=q_embedding,
            n_results=min(top_k, len(RECIPES)),
            where={"name": {"$ne": "metadata"}}   # exclude __meta__ row
        )

        # ChromaDB returns cosine DISTANCE (0=identical, 2=opposite)
        # Convert to SIMILARITY: similarity = 1 - distance
        pairs = [
            (rid, round(1.0 - dist, 4))
            for rid, dist in zip(results["ids"][0], results["distances"][0])
        ]
        log.debug(f"Semantic search '{query[:40]}' → top: {pairs[:3]}")
        return pairs

    except Exception as e:
        log.warning(f"Semantic search error: {e}")
        return []

def keyword_score(recipe: dict, user_ings: list[str], query: str) -> float:
    """Keyword matching component of the hybrid score."""
    q   = norm(query)
    rc  = recipe["_ci"]
    tags= set(recipe.get("tags", []))

    if not user_ings:
        # Name word overlap
        nw = [w for w in norm(recipe["name"]).split() if len(w) > 2]
        return sum(1 for w in nw if w in q) / max(len(nw), 1)

    us = set(user_ings)
    # Seafood expansion
    if us & SEAFOOD and recipe["_sf"]:
        for sf in SEAFOOD:
            if sf in rc: us = us | {sf}

    matched = {u for u in us if u in rc or any(len(u) > 2 and (u in r or r in u) for r in rc)}
    overlap  = len(matched) / max(len(rc), 1)
    # Name boost: ingredient appears in recipe name
    nb = min(0.3, sum(0.15 for u in us if u in norm(recipe["name"])))
    # Multi-ingredient bonus
    multi = min(0.2, (len(matched) - 1) * 0.06) if len(matched) > 1 else 0.0
    # Tag/cuisine match
    tb = sum(0.04 for t in tags if t in q) + (0.05 if recipe.get("cuisine") and norm(recipe["cuisine"]) in q else 0)
    return min(1.0, overlap * 0.7 + nb * 0.2 + multi + tb)

def hybrid_score(sem: float, kw: float) -> float:
    """Hybrid = 0.7 * semantic + 0.3 * keyword (semantic-first)."""
    return round(0.70 * sem + 0.30 * kw, 4)

def apply_hard_filters(
    recipe: dict, exclude: list[str],
    diet_filter: Optional[str], meal_type: Optional[str], quick: bool
) -> bool:
    """Returns True if recipe PASSES all hard filters."""
    rc   = recipe["_ci"]
    rdiet= set(recipe.get("diet", []))
    tags = set(recipe.get("tags", []))

    # Diet hard filter
    if diet_filter == "vegan"       and "vegan" not in rdiet: return False
    if diet_filter == "vegetarian"  and not (rdiet & {"vegetarian","vegan"}): return False
    if diet_filter == "non-veg"     and "non-veg" not in rdiet: return False
    if diet_filter == "pescatarian" and not (rdiet & {"pescatarian","vegetarian","vegan"}): return False

    # Exclusion hard filter
    excl = set(exclude)
    for ex in excl:
        if ex in rc or any(len(ex) > 3 and (ex in r or r in ex) for r in rc): return False
    if "seafood" in excl and recipe["_sf"]: return False
    for sf in SEAFOOD:
        if sf in excl and (sf in rc or any(sf in r for r in rc)): return False

    # Quick filter
    if quick:
        try:
            mins = int(re.search(r'(\d+)', recipe.get("time", "60")).group(1))
            if mins > 30: return False
        except: pass

    # Meal-type hard filter
    HARD_MEAL = {"dessert","breakfast","soup","drink","salad","snack"}
    if meal_type and meal_type in HARD_MEAL:
        matched_meal = (
            meal_type in tags or
            any(meal_type in t for t in tags) or
            (meal_type == "dessert" and tags & {"sweet","baked","frozen","no-bake","cheesecake","halwa"}) or
            (meal_type == "breakfast" and tags & {"egg","quick","sweet"}) or
            (meal_type == "drink" and tags & {"smoothie","lassi","juice","drink"}) or
            (meal_type == "soup" and "soup" in norm(recipe.get("name","")))
        )
        if not matched_meal: return False

    return True

def rag_search(
    query: str,
    user_ings: list[str],
    exclude: list[str],
    diet_filter: Optional[str] = None,
    meal_type: Optional[str] = None,
    quick: bool = False,
    top_k: int = 5
) -> list[dict]:
    """
    TRUE RAG PIPELINE:
      Step 1: Semantic retrieval via ChromaDB (cosine similarity)
      Step 2: Hard filter (exclusions, diet, meal-type)
      Step 3: Compute keyword score for retrieved candidates
      Step 4: Hybrid rerank (0.7 sem + 0.3 kw)
      Step 5: Return top-K
    """
    # ── STEP 1: Semantic retrieval ──────────────────────────────────────
    # Enrich query with canonicalized ingredients for better embedding match
    enriched_query = query
    if user_ings:
        enriched_query = f"{query}. Ingredients: {' '.join(user_ings)}"
    if meal_type:
        enriched_query += f". Meal type: {meal_type}"
    if diet_filter:
        enriched_query += f". Diet: {diet_filter}"

    sem_pairs = semantic_search(enriched_query, top_k=min(30, len(RECIPES)))

    # Build recipe lookup
    recipe_map = {r["id"]: r for r in RECIPES}

    # ── STEP 2: Hard filter ─────────────────────────────────────────────
    filtered = []
    for rid, sem_sim in sem_pairs:
        if rid not in recipe_map: continue
        recipe = recipe_map[rid]
        if apply_hard_filters(recipe, exclude, diet_filter, meal_type, quick):
            filtered.append((recipe, sem_sim))

    # Fallback: if semantic retrieval + filters yield nothing, try all recipes
    if not filtered:
        log.info("Semantic retrieval empty after filters — scanning all recipes")
        for r in RECIPES:
            if apply_hard_filters(r, exclude, diet_filter, meal_type, quick):
                # Use keyword score only
                ks = keyword_score(r, user_ings, query)
                if ks > 0.01:
                    filtered.append((r, 0.0))

    # ── STEP 3+4: Compute keyword score and hybrid rerank ───────────────
    results = []
    for recipe, sem_sim in filtered:
        ks    = keyword_score(recipe, user_ings, query)
        score = hybrid_score(sem_sim, ks)

        # Compute matched / missing ingredients
        rc = recipe["_ci"]
        us = set(user_ings)
        if us & SEAFOOD and recipe["_sf"]:
            for sf in SEAFOOD:
                if sf in rc: us = us | {sf}
        matched = sorted(u for u in us if u in rc or any(len(u) > 2 and (u in r2 or r2 in u) for r2 in rc))
        missing = sorted(rc - set(user_ings))[:6]
        mp      = min(99, round(100 * len(matched) / max(len(rc), 1))) if user_ings else None

        results.append({
            "recipe":    recipe,
            "sem_score": sem_sim,
            "kw_score":  ks,
            "score":     score,
            "matched":   matched,
            "missing":   missing,
            "match_pct": mp,
        })

    # Sort by hybrid score
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

# ─── GROQ HELPERS ─────────────────────────────────────────────────────────────
async def _groq(messages: list, max_tokens=400, temp=0.0) -> str:
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={"model": GROQ_MDL, "messages": messages, "max_tokens": max_tokens, "temperature": temp}
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

NLU_SYS = """You are an intent extractor for a Recipe Assistant. Return ONLY valid JSON:
{"intent":"find_recipe|substitution|set_preference|override_preference|confirm_gen|decline_gen|random_recipe|greeting|help|thanks","ingredients":[],"exclude":[],"diet":[],"meal_type":null,"cuisine":null,"query_for_search":""}
Rules:
- "no X"/"without X"/"avoid X"/"don't want X" → exclude:[X], intent=set_preference
- "include X"/"add X back"/"use X again" → intent=override_preference
- "reset preferences"/"clear filters" → override_preference, ingredients:["__reset__"]
- "I have X,Y,Z" → ingredients:[X,Y,Z]
- "yes/sure/ok/generate/go ahead/yep/absolutely" (when pending generation) → confirm_gen
- "no/skip/cancel/never mind" → decline_gen
- "random/surprise me" → random_recipe
- "veg/vegetarian" → diet:["vegetarian"] | "vegan" → diet:["vegan"] | "non-veg/meat" → diet:["non-veg"]
- meal_type: breakfast|lunch|dinner|dessert|drink|snack|soup|salad
- query_for_search: clean rewritten search phrase for semantic embedding e.g. "spicy chicken dinner recipe"
Return ONLY the JSON object, nothing else."""

async def groq_nlu(text: str, history: list) -> dict:
    msgs = [{"role": "system", "content": NLU_SYS}]
    for h in history[-4:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": text})
    try:
        raw = await _groq(msgs, max_tokens=300)
        raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception as e:
        log.warning(f"Groq NLU failed: {e} — using rule-based fallback")
        return _rule_nlu(text)

def _rule_nlu(text: str) -> dict:
    t = norm(text); intent = "find_recipe"
    if re.search(r"^(hi|hello|hey)\b", t): intent = "greeting"
    elif re.search(r"\b(help|what can you)\b", t): intent = "help"
    elif re.search(r"\b(thank|awesome|great)\b", t): intent = "thanks"
    elif re.search(r"\b(substitute|replace|instead of|alternative)\b", t): intent = "substitution"
    elif re.search(r"\b(reset|clear|forget).*(prefer|exclude)\b|\breset preferences\b", t):
        return {"intent":"override_preference","ingredients":["__reset__"],"exclude":[],"diet":[],"meal_type":None,"cuisine":None,"query_for_search":text}
    elif re.search(r"\b(include|add back)\s+\w", t): intent = "override_preference"
    elif re.search(r"\bno\s+\w|\bwithout\s+\w|\bavoid\s+\w|\bdon.?t\s+(eat|want)\s+\w", t) and not re.search(r"\b(recipe|dish|cook)\b", t): intent = "set_preference"
    elif re.search(r"\b(random|surprise)\b", t): intent = "random_recipe"
    elif re.search(r"^(yes|sure|ok|go ahead|generate|yeah|please|yep|yup|absolutely)\b", t): intent = "confirm_gen"
    elif re.search(r"^(no|nope|skip|cancel|never mind)\b", t): intent = "decline_gen"
    excl = [m.group(2) for m in re.finditer(r"\b(no|without|avoid|exclude)\s+(\w+)", t)]
    override = [m.group(2) for m in re.finditer(r"\b(include|add back|use)\s+(\w+)", t)]
    dd = detect_diet_filter(t); dd = [dd] if dd else []
    meal = detect_meal(t)
    cuisine = next((c for c in ["indian","italian","chinese","thai","mexican","american","mediterranean"] if c in t), None)
    return {"intent": intent, "ingredients": extract_ingredients(t) + override, "exclude": excl, "diet": dd,
            "meal_type": meal, "cuisine": cuisine.title() if cuisine else None, "query_for_search": text}

async def groq_llm_rerank(user_ings: list, query: str, candidates: list[dict], sess: dict) -> list[dict]:
    """LLM re-ranks top candidates based on reasoning (Google-style)."""
    if len(candidates) <= 1: return candidates
    lines = "\n".join(
        f"{i+1}.[{h['recipe']['id']}] {h['recipe']['name']} | sem={h['sem_score']:.3f} | kw={h['kw_score']:.3f} | diet:{h['recipe']['diet']} | ings:{', '.join(list(h['recipe'].get('ingredients',{}).keys())[:5])}"
        for i, h in enumerate(candidates[:10])
    )
    prompt = (
        f"Rank these recipes by best match for the user.\n"
        f"User query: \"{query}\"\n"
        f"User ingredients: {user_ings or 'not specified'}\n"
        f"EXCLUDED (must NOT appear in top): {sess.get('exclude', [])}\n"
        f"Diet filter: {sess.get('diet', [])}\n\n"
        f"Candidates (id, name, semantic score, keyword score, diet, top ingredients):\n{lines}\n\n"
        f"Return ONLY a JSON array of recipe IDs in ranked order. Example: [\"r07\",\"r12\",\"r08\"]\n"
        f"RULES: recipes with excluded ingredients must be ranked last. Diet-matching recipes first."
    )
    try:
        raw = await _groq([{"role": "user", "content": prompt}], max_tokens=150)
        raw = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
        ordered_ids = json.loads(raw)
        id_map = {h["recipe"]["id"]: h for h in candidates}
        reranked = [id_map[rid] for rid in ordered_ids if rid in id_map]
        # Append anything not in LLM output
        seen = set(ordered_ids)
        for h in candidates:
            if h["recipe"]["id"] not in seen: reranked.append(h)
        return reranked
    except Exception as e:
        log.warning(f"LLM rerank failed: {e}")
        return candidates

async def groq_explain(recipe: dict, user_ings: list, matched: list, sem_score: float) -> str:
    """Generate conversational explanation of why this recipe was selected."""
    try:
        prompt = (
            f"Recipe: {recipe['name']} ({recipe['cuisine']}, {recipe['time']}).\n"
            f"User has: {user_ings or 'general query'}. Matched ingredients: {matched}.\n"
            f"Semantic similarity score: {sem_score:.2f}/1.0.\n"
            f"Write ONE warm, natural sentence (max 22 words) explaining why this is the top match. No markdown, no lists."
        )
        return await _groq([{"role": "user", "content": prompt}], max_tokens=70, temp=0.3)
    except Exception:
        n = len(matched)
        return f"Uses {n} of your ingredient{'s' if n!=1 else ''} with {sem_score:.0%} semantic similarity — {'great' if n>=3 else 'good'} match!"

async def groq_generate_recipe(query: str, exclude: list, diet: list) -> Optional[dict]:
    """Generate a completely new recipe using LLM when database has no match."""
    prompt = (
        f'Create a complete, realistic recipe for: "{query}".\n'
        f'Excluded ingredients: {exclude or "none"}.\n'
        f'Diet requirement: {diet or "any"}.\n'
        f'Return ONLY valid JSON (no markdown):\n'
        f'{{"name":"Recipe Name","cuisine":"Cuisine","diet":["vegetarian"],"time":"30 min","servings":4,'
        f'"tags":["tag1","tag2"],"ingredients":{{"ingredient":"amount"}},'
        f'"steps":["Detailed step 1.","Detailed step 2.","Detailed step 3.","Detailed step 4.","Detailed step 5."]}}'
    )
    for attempt in range(3):
        try:
            raw  = await _groq([{"role": "user", "content": prompt}], max_tokens=1000, temp=0.6)
            raw  = re.sub(r"^```json|^```|```$", "", raw, flags=re.MULTILINE).strip()
            m    = re.search(r'\{.*\}', raw, re.DOTALL)
            if m: raw = m.group(0)
            data = json.loads(raw)
            if "name" in data and "ingredients" in data and "steps" in data:
                return data
        except Exception as e:
            log.warning(f"Generate attempt {attempt+1}: {e}")
            if attempt < 2: await asyncio.sleep(1)
    return None

def find_substitution(text: str, nlu: dict) -> Optional[str]:
    for ing in (nlu.get("ingredients") or []):
        c = canonicalize(norm(ing))
        if c in SUBS: return c
    t = norm(text)
    for pat in [
        r"substitute (?:for )?([a-z ]+?)(?:\?|$|,|\band\b)",
        r"replace (?:the )?([a-z ]+?)(?:\?|$|,)",
        r"instead of ([a-z ]+?)(?:\?|$|,)",
        r"without ([a-z ]+?)(?:\?|$|,)",
        r"alternative (?:to|for) ([a-z ]+?)(?:\?|$)",
    ]:
        m = re.search(pat, t)
        if m:
            c = canonicalize(m.group(1).strip())
            if c in SUBS: return c
    return next((k for k in sorted(SUBS, key=len, reverse=True) if k in t), None)

# ─── MAIN HANDLER ─────────────────────────────────────────────────────────────
def make_card(h: dict) -> dict:
    r = h["recipe"]
    return {"id": r["id"], "name": r["name"], "cuisine": r["cuisine"], "diet": r["diet"],
            "time": r["time"], "servings": r["servings"], "tags": r.get("tags", []),
            "ingredients": r.get("ingredients", {}), "steps": r.get("steps", []),
            "match_pct": h["match_pct"], "missing": h["missing"],
            "matched_ingredients": h.get("matched", []),
            "sem_score": h.get("sem_score", 0), "_is_llm_generated": r.get("_is_llm_generated", False)}

async def handle(text: str, sess: dict) -> dict:
    nlu    = await groq_nlu(text, sess["history"])
    intent = nlu.get("intent", "find_recipe")

    # Check pending generation — broaden intent detection
    if sess.get("pending_gen") and intent not in ("decline_gen","override_preference","greeting","help","substitution"):
        t = norm(text)
        if re.search(r'^(yes|sure|ok|yeah|go|please|generate|create|do it|yep|yup|absolutely|make it)\b', t):
            intent = "confirm_gen"
        elif re.search(r'^(no|nope|skip|cancel|never mind)\b', t):
            intent = "decline_gen"

    # ── Confirm AI generation ──────────────────────────────────────────────
    if intent == "confirm_gen":
        pg = sess.get("pending_gen"); sess["pending_gen"] = None
        if not pg:
            return {"type":"error","message":"Please tell me what recipe to generate first!","recipes":[],"sub":None}
        log.info(f"Generating recipe for: {pg['query']}")
        gen = await groq_generate_recipe(pg["query"], sess.get("exclude",[]), sess.get("diet",[]))
        if gen:
            gen["id"] = "llm_gen"; gen["_is_llm_generated"] = True
            gen["_ci"] = set(); gen["_sf"] = False
            card = make_card({"recipe": gen, "match_pct": None, "missing": [], "matched": [], "sem_score": 1.0})
            return {"type":"recipes","message":f"✨ **AI-Generated: {gen.get('name','Custom Recipe')}**\n\nCreated just for you based on your request!","recipes":[card],"sub":None}
        return {"type":"error","message":"⚠️ Generation failed. Please rephrase, e.g. *'generate a strawberry dessert recipe'*","recipes":[],"sub":None}

    if intent == "decline_gen":
        sess["pending_gen"] = None
        return {"type":"decline","message":"👍 No problem! Try different ingredients or say **'reset preferences'** to clear filters.","recipes":[],"sub":None}

    # ── Preference override ────────────────────────────────────────────────
    if intent == "override_preference":
        ings = nlu.get("ingredients", [])
        if "__reset__" in ings:
            sess["exclude"] = []; sess["diet"] = []; sess["pending_gen"] = None
            return {"type":"pref","message":"✅ All preferences reset! Fresh start — what would you like to cook?","recipes":[],"sub":None}
        restored = []
        for ing in ings:
            c = canonicalize(norm(ing))
            if c in sess["exclude"]: sess["exclude"].remove(c); restored.append(c)
            elif ing in sess["exclude"]: sess["exclude"].remove(ing); restored.append(ing)
        msg = f"✅ Removed **{', '.join(restored)}** from exclusions. Showing results with it now!" if restored else "✅ Preferences updated."
        return {"type":"pref","message":msg,"recipes":[],"sub":None}

    # Merge preferences
    for e in nlu.get("exclude", []):
        c = canonicalize(norm(e))
        if c and c not in sess["exclude"]: sess["exclude"].append(c)
    for d in nlu.get("diet", []):
        if d not in sess["diet"]: sess["diet"].append(d)

    # ── Simple intents ─────────────────────────────────────────────────────
    if intent == "greeting":
        return {"type":"greeting","message":"👋 Hey! I'm your AI Recipe Assistant powered by RAG (semantic search + embeddings).\n\n• **'I have chicken, rice'** — find recipes by ingredients\n• **'veg dinner'** / **'no beef'** — dietary filters\n• **'include chicken'** / **'reset preferences'** — undo filters\n• **'substitute for paneer'** — ingredient alternatives\n• **'random recipe'** — surprise me!\n• Upload a photo 📷 to scan ingredients","recipes":[],"sub":None}
    if intent == "help":
        return {"type":"help","message":"🍳 **RAG-Powered Recipe Search:**\n• Semantic embedding search (ChromaDB + MiniLM)\n• Hybrid score: 70% semantic + 30% keyword match\n• LLM re-ranking via Groq LLaMA-3 70B\n• AI recipe generation when no match found\n• Hard dietary filters + ingredient exclusions\n• OCR ingredient scanning from photos","recipes":[],"sub":None}
    if intent == "thanks":
        return {"type":"thanks","message":"😊 Happy cooking! Ask me anything.","recipes":[],"sub":None}
    if intent == "substitution":
        t = find_substitution(text, nlu)
        if t: return {"type":"sub","message":f"🔄 Substitutes for **{t}**:","recipes":[],"sub":{"ingredient":t,"options":SUBS[t]}}
        return {"type":"sub","message":"Couldn't identify the ingredient. Try: *'substitute for paneer'*","recipes":[],"sub":None}
    if intent == "set_preference" and not nlu.get("ingredients"):
        parts = ([f"excluding **{', '.join(sess['exclude'])}**"] if sess["exclude"] else []) + ([f"**{', '.join(sess['diet'])}** diet"] if sess["diet"] else [])
        return {"type":"pref","message":"✅ Preferences: " + (", ".join(parts) or "none") + ".\nSay *'reset preferences'* or *'include [ingredient]'* to undo.","recipes":[],"sub":None}
    if intent == "random_recipe":
        df = detect_diet_filter(norm(text)) or (sess["diet"][0] if sess["diet"] else None)
        excl_s = set(sess.get("exclude",[]))
        def passes(r):
            if any(e in r["_ci"] for e in excl_s): return False
            rd = set(r.get("diet",[]))
            if df=="vegan" and "vegan" not in rd: return False
            if df=="vegetarian" and not(rd&{"vegetarian","vegan"}): return False
            if df=="non-veg" and "non-veg" not in rd: return False
            return True
        av = [r for r in RECIPES if passes(r)] or RECIPES
        picks = random.sample(av, min(5, len(av)))
        dl = f" **{df}**" if df else ""
        return {"type":"recipes","message":f"🎲 Here are **{len(picks)} random{dl} recipes** to inspire you!",
                "recipes":[make_card({"recipe":r,"match_pct":None,"missing":[],"matched":[],"sem_score":0}) for r in picks],"sub":None}

    # ── MAIN RAG SEARCH ────────────────────────────────────────────────────
    nlu_ings  = [canonicalize(norm(i)) for i in (nlu.get("ingredients") or [])]
    rule_ings = extract_ingredients(text)
    user_ings = list(dict.fromkeys(nlu_ings + [i for i in rule_ings if i not in nlu_ings]))

    meal_type  = nlu.get("meal_type") or detect_meal(text)
    quick      = any(k in norm(text) for k in QUICK_KW)
    sq         = nlu.get("query_for_search") or text

    # Diet from NLU + session
    all_diets  = list(dict.fromkeys((nlu.get("diet") or []) + sess.get("diet", [])))
    df_from    = detect_diet_filter(norm(text))
    diet_filter = df_from or (all_diets[0] if all_diets else None)
    if diet_filter and diet_filter not in sess["diet"]: sess["diet"].append(diet_filter)

    # Build combined exclusion list
    combined_exclude = list(dict.fromkeys(
        [canonicalize(norm(e)) for e in (nlu.get("exclude") or [])]
        + [canonicalize(norm(e)) for e in sess.get("exclude", [])]
    ))

    # ── RAG retrieval ──────────────────────────────────────────────────────
    hits = rag_search(
        query        = sq,
        user_ings    = user_ings,
        exclude      = combined_exclude,
        diet_filter  = diet_filter,
        meal_type    = meal_type,
        quick        = quick,
        top_k        = 10
    )

    # ── No results → offer AI generation ──────────────────────────────────
    if not hits:
        sess["pending_gen"] = {"query": text, "ingredients": user_ings}
        excl_s  = f" excluding **{', '.join(sess['exclude'])}**" if sess.get("exclude") else ""
        meal_h  = f" for **{meal_type}**" if meal_type else ""
        diet_h  = f" (**{diet_filter}**)" if diet_filter else ""
        return {
            "type":    "no_result",
            "message": f"😔 No matching recipes found{diet_h}{meal_h}{excl_s}.\n\n✨ **Want me to generate a custom AI recipe for this?** Just say **yes**!",
            "recipes": [], "sub": None
        }

    # ── LLM re-ranking ─────────────────────────────────────────────────────
    hits = await groq_llm_rerank(user_ings, sq, hits, sess)

    # ── Final hard-filter pass after LLM rerank ────────────────────────────
    def passes_final(h):
        r  = h["recipe"]; rc = r["_ci"]; rd = set(r.get("diet",[]))
        ex = set(combined_exclude)
        if any(e in rc or any(e in c or c in e for c in rc if len(e)>3) for e in ex): return False
        if diet_filter=="vegetarian" and not(rd&{"vegetarian","vegan"}): return False
        if diet_filter=="vegan" and "vegan" not in rd: return False
        if diet_filter=="non-veg" and "non-veg" not in rd: return False
        return True
    filtered = [h for h in hits if passes_final(h)]
    final = (filtered or hits)[:5]
    top   = final[0]

    explanation = await groq_explain(top["recipe"], user_ings, top.get("matched",[]), top.get("sem_score",0))
    ing_str = f" using **{', '.join(user_ings[:4])}**" if user_ings else ""
    pct_str = f" ({top['match_pct']}% ingredient match)" if top["match_pct"] is not None else ""
    sem_str = f" | {top['sem_score']:.0%} semantic similarity"

    msg = f"🍽️ Found **{len(final)} recipe{'s' if len(final)>1 else ''}**{ing_str}{pct_str}{sem_str}\n\n💡 {explanation}"
    if top["missing"]:
        msg += f"\n\n⚠️ Top recipe also needs: **{', '.join(top['missing'][:4])}**"
    if sess.get("exclude"):
        msg += f"\n✅ Excluded: **{', '.join(sess['exclude'])}**"
    if diet_filter:
        msg += f"\n🥗 Filter: **{diet_filter}** only"

    return {"type":"recipes","message":msg,"recipes":[make_card(h) for h in final],"sub":None}

# ─── OCR ──────────────────────────────────────────────────────────────────────
def _ocr_extract(img_bytes: bytes) -> list[str]:
    try:
        import easyocr, numpy as np
        from PIL import Image, ImageEnhance
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = ImageEnhance.Contrast(ImageEnhance.Sharpness(img).enhance(2.0)).enhance(1.5)
        reader  = easyocr.Reader(["en"], gpu=False, verbose=False)
        results = reader.readtext(np.array(img), detail=0, paragraph=True)
        text    = " ".join(results)
        log.info(f"EasyOCR raw: {text[:120]}")
        ings = extract_ingredients(text)
        if ings: return ings
    except Exception as e:
        log.warning(f"EasyOCR failed: {e}")
    try:
        from PIL import Image, ImageFilter, ImageEnhance
        import pytesseract
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        img = ImageEnhance.Contrast(
            img.resize((img.width*2, img.height*2), Image.LANCZOS).filter(ImageFilter.SHARPEN)
        ).enhance(2.5)
        text = pytesseract.image_to_string(img, config="--psm 6 --oem 3")
        log.info(f"Pytesseract raw: {text[:120]}")
        return extract_ingredients(text)
    except Exception as e:
        log.warning(f"Pytesseract failed: {e}")
    return []

# ─── FASTAPI ROUTES ───────────────────────────────────────────────────────────
class ChatReq(BaseModel):
    session_id: str = ""
    message: str

@app.on_event("startup")
async def startup():
    # Build ChromaDB embeddings in background (non-blocking)
    asyncio.create_task(asyncio.to_thread(build_chroma))

@app.post("/chat")
async def chat(req: ChatReq):
    sid  = req.session_id or str(uuid.uuid4())
    sess = get_sess(sid)
    sess["history"].append({"role": "user", "content": req.message})
    result = await handle(req.message, sess)
    sess["history"].append({"role": "assistant", "content": result["message"]})
    return {**result, "session_id": sid, "preferences": {"exclude": sess["exclude"], "diet": sess["diet"]}}

@app.post("/ocr")
async def ocr(file: UploadFile = File(...), session_id: str = Form(default="default")):
    detected = await asyncio.to_thread(_ocr_extract, await file.read())
    if not detected:
        return {"session_id": session_id, "detected": [],
                "message": "⚠️ No ingredients detected. Please try a clearer photo or type them.", "recipes": []}
    sess   = get_sess(session_id)
    result = await handle("recipes with " + " ".join(detected), sess)
    return {"session_id": session_id, "detected": detected,
            "message": f"📸 Detected: **{', '.join(detected)}**\n\n" + result["message"],
            "recipes": result["recipes"]}

@app.post("/reset")
async def reset(body: dict):
    sid = body.get("session_id", "")
    if sid in SESSIONS: del SESSIONS[sid]
    return {"status": "reset"}

@app.get("/health")
def health():
    return {
        "status":    "ok",
        "version":   "RAG v7",
        "recipes":   len(RECIPES),
        "chroma":    "ready" if _col else "loading",
        "embed_model": EMBED_MODEL if _col else "pending",
        "subs":      len(SUBS),
        "groq":      "configured",
        "pipeline":  "semantic(0.7) + keyword(0.3) + LLM-rerank"
    }

@app.get("/")
def home():
    return {"status":"online","engine":"Recipe RAG v7","pipeline":"ChromaDB cosine → hybrid rerank → Groq LLaMA"}

if __name__ == "__main__":
    import sys
    if "--eval" in sys.argv:
        async def _eval():
            print("\n🧪 RAG v7 — Accuracy Evaluation")
            cases = [
                ("chicken and rice", ["chicken"], []),
                ("recipe using fish", ["fish","salmon","shrimp","cod","tuna"], []),
                ("recipe using prawns", ["shrimp"], []),
                ("apple dessert", ["apple"], []),
                ("strawberry cheesecake", ["strawberry"], []),
                ("carrot dessert", ["carrot"], []),
                ("no beef pasta", ["pasta"], ["beef"]),
                ("cauliflower soup", ["cauliflower"], []),
                ("vegetarian Indian curry", ["chickpeas","paneer","lentils","potato"], ["chicken","beef"]),
                ("quick dinner with eggs", ["eggs"], []),
                ("mango drink", ["mango"], []),
                ("paneer dishes", ["paneer"], []),
                ("chocolate cake", ["chocolate"], []),
                ("salmon recipe", ["salmon"], []),
            ]
            passed = 0
            for q, must, must_not in cases:
                user_ings = extract_ingredients(q)
                diet = detect_diet_filter(norm(q))
                meal = detect_meal(q)
                quick = any(k in norm(q) for k in QUICK_KW)
                hits = rag_search(q, user_ings, must_not, diet, meal, quick, top_k=5)
                ok = True
                if not hits:
                    print(f"  ❌ FAIL (no results): '{q}'"); continue
                top_r = hits[0]["recipe"]
                top_ci = top_r["_ci"] | {norm(k) for k in top_r.get("ingredients",{})}
                top_tags = set(top_r.get("tags",[]))
                for m in must:
                    mc = canonicalize(norm(m))
                    hit = (mc in top_ci or m in top_ci or m in top_tags or
                           (m in SEAFOOD and top_r["_sf"]) or
                           any(m in c or c in m for c in top_ci if len(m)>2))
                    if not hit:
                        ok = False
                        print(f"  ❌ FAIL: '{q}' → {top_r['name']!r} missing '{m}'"); break
                for mn in must_not:
                    mnc = canonicalize(norm(mn))
                    if mnc in top_ci or any(mn in c or c in mn for c in top_ci if len(mn)>2):
                        ok = False; print(f"  ❌ FAIL: '{q}' — excluded '{mn}' in top result")
                if ok:
                    passed += 1
                    print(f"  ✅ PASS: '{q}' → {top_r['name']!r} (sem={hits[0]['sem_score']:.3f})")
            pct = 100*passed//len(cases)
            print(f"\n📊 {passed}/{len(cases)} = {pct}% accuracy")
            print("🎯 Target achieved!" if pct >= 80 else "⚠️  Below target.")
        asyncio.run(_eval())
    else:
        uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
