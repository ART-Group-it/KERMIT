var result = []; //is the data structure final

//Start to divide the parse tree and put into a data structure
var res = string.slice(string.indexOf('('), string.lastIndexOf("'"));
var pilaParent = [];
var firstIndex = res.indexOf(':');
var j = 0;
for (i = 0; i <= res.length; i++) {

    if (res[i] == '(') {
        var nome = computeName(i);
        i = i + nome.length;
        var valore = computeValue(i + 1);
        i = i + valore.length;

        if (j == 0) {
            result[j] = {
                name: nome,
                value: valore,
                parent: {},
                id: j,
                disegnato: false
            }
            pilaParent.push(result[j]);
        } else {
            result[j] = {
                name: nome,
                value: valore,
                parent: {
                    padre: pilaParent[pilaParent.length - 1]
                },
                id: j,
                disegnato: false
            }
            pilaParent.push(result[j]);
        }
        j++;
    }


    if (res[i] == ' ' && res[i + 1] != '(') {
        var nome = computeName(i);
        i = i + nome.length;
        var valore = computeValue(i + 1);
        i = i + valore.length;

        result[j] = {
            name: nome,
            value: valore,
            parent: {
                padre: pilaParent[pilaParent.length - 1]
            },
            id: j,
            disegnato: false
        }
        j++;
    }

    if (res[i] == ')') {
        pilaParent.pop();
    }
}


for (i = 0; i < result.length; i++) {
    var nodo = result[i].id;
    var figli = [];
    for (j = i; j < result.length; j++) {
        if (j == 0) {
            j++;
        }
        if (nodo == result[j].parent.padre.id) {
            figli.push(result[j]);
        }
    }
    result[i].children = figli;
}

function computeName(index) {  //function that computes the name of each node
    var primaDuePunti = res.indexOf(":", index)

    if (res[primaDuePunti + 1] == res[primaDuePunti]) {
        return res.slice(index + 1, primaDuePunti + 1);
    } else {
        return res.slice(index + 1, primaDuePunti);
    }
}

function computeValue(index) { //function that computes the value of each node
    var inizioNumeri = index + 1;
    var number = '';
    while ((res[inizioNumeri] != ' ') && (res[inizioNumeri] != ')') && (inizioNumeri < res.length)) {
        number = number + res[inizioNumeri];
        inizioNumeri++;
    }
    if (isNaN(number)) {
        var ultimiDuePunti = number.lastIndexOf(":") + 1;
        number = number.slice(ultimiDuePunti);
    }
    return number;
}


var livello = 0;

function calcolaAltezza(dataStructure, level) {  //function that computes the height of the tree

    var numChild = dataStructure.children.length;
    dataStructure.livello = level;
    if (numChild == 0) {
        return;
    }
    for (let i = 0; i < dataStructure.children.length; i++) {
        calcolaAltezza(dataStructure.children[i], level + 1)
    }
    level = level + 1;
    if (level > livello) {
        livello = level;
    }
}
calcolaAltezza(result[0], 1)



var arraySpazioLivelli = [];
for (let i = 0; i < livello; i++) {
    arraySpazioLivelli[i] = 0;
}
var spazioOccupato = 0;

//initialize the canvas
var canvas = document.getElementById('myCanvas');
canvas.height = levelLength * (livello + 1);
canvas.width = window.innerWidth;


//function that take in input the datastructure resulting from the parse tree and the level and draws the entire Tree
function drawTree(dataStructure, level) {
    var daDisegnare = true;
    var numChild = dataStructure.children.length;

    for (let i = 0; i < numChild; i++) {
        drawTree(dataStructure.children[i], level + 1);
    }

    if (numChild == 0) { //base case
        var spazioNodo = dataStructure.name.length * (4.3 + (parseFloat(dataStructure.value) * 10))
        if (level > 0 && arraySpazioLivelli[level - 2] > arraySpazioLivelli[level - 1]) {
            arraySpazioLivelli[level - 1] = arraySpazioLivelli[level - 2];
        }
        drawNode(dataStructure, arraySpazioLivelli[level - 1], level * levelLength);
        arraySpazioLivelli[level - 1] += spazioNodo + brotherDistance;
        for (let i = level - 1; i < arraySpazioLivelli.length; i++) {
            if (arraySpazioLivelli[i] < (arraySpazioLivelli[level - 1])) {
                arraySpazioLivelli[i] = (arraySpazioLivelli[level - 1]);
            }
        }
        daDisegnare = false;
    }

    for (let i = 0; i < dataStructure.children.length; i++) {
        if (dataStructure.children[i].disegnato == false) {
            daDisegnare = false;
        }
    }

    if (daDisegnare == true) {
        if (dataStructure.children.length == 1) {
            if (dataStructure.children[0].children.length == 0) {
                var spazioNodo = dataStructure.name.length * (4.3 + (parseFloat(dataStructure.value) * 10))
                drawNode(dataStructure, dataStructure.children[0].x, level * levelLength);
                arraySpazioLivelli[level - 1] = dataStructure.children[0].x + spazioNodo + brotherDistance;
            } else {
                var spazioNodo = dataStructure.name.length * (4.3 + (parseFloat(dataStructure.value) * 10))
                var spazioNodoFiglio = dataStructure.children[0].name.length * (4.3 + (parseFloat(dataStructure.value) * 10))
                drawNode(dataStructure, arraySpazioLivelli[level - 1], level * levelLength);
                arraySpazioLivelli[level - 1] += spazioNodo + brotherDistance;
            }
        } else {
            var XprimoFiglio = dataStructure.children[0].x;
            var XultimoFiglio = dataStructure.children[dataStructure.children.length - 1].x;
            var spazioNodo = dataStructure.name.length * (4.3 + (parseFloat(dataStructure.value) * 10))
            drawNode(dataStructure, arraySpazioLivelli[level - 1], level * levelLength);
            arraySpazioLivelli[level - 1] += spazioNodo + brotherDistance;
        }
    }
}

drawTree(result[0], 1);
drawBranch(result[0], 1);
canvas.width = arraySpazioLivelli[arraySpazioLivelli.length - 1];
for (let i = 0; i < livello; i++) {
    arraySpazioLivelli[i] = 0;
}
drawTree(result[0], 1);
drawBranch(result[0], 1);



function drawNode(node, center, height) { //function that draws the node node in posizion center, height
    node.x = center;
    node.y = height;
    node.disegnato = true;
    var ctx = canvas.getContext("2d");
    ctx.font = 10 + Math.round(node.value * 10) + "px" + "  monospace";
    ctx.fillStyle = perc2color(node.value);
    ctx.imageSmoothingEnabled = false;
    ctx.fillText(node.name, center, height);
}

function drawBranch(node, level) {  //function that takes in input the node and the respective level and draws the branch to the node childrens

    var numChild = node.children.length;

    for (let i = 0; i < numChild; i++) {
        drawBranch(node.children[i], level + 1);
        var ctx = canvas.getContext("2d");
        ctx.strokeStyle = perc2color(node.children[i].value);
        ctx.imageSmoothingEnabled = false;
        ctx.beginPath();
        if (node.children[i].children.length == 0) {
            ctx.moveTo(node.x + 4.3, node.y + 3);
            ctx.lineTo(node.x + 4.3, node.children[i].y - 15);
        } else {
            var spazioNodo = node.children[i].name.length * (4.3 + (parseFloat(node.children[i].value) * 10))
            var spazioNodoPadre = node.name.length * (4.3 + (parseFloat(node.value) * 10))
            ctx.moveTo(node.x + spazioNodoPadre / 2, node.y + 3);
            ctx.lineTo(node.children[i].x + spazioNodo / 2, node.children[i].y - 15)
        }
        ctx.closePath();
        ctx.stroke();
    }
}

function isEmpty(obj) {
    return Object.keys(obj).length === 0;
}

function perc2color(perc) {  //function that assigns the correct color according to the value
    perc = perc * 100;
    var r, g, b = 0;
    if (perc < 50) {
        r = Math.round(5.1 * perc);
        g = Math.round(5.1 * perc);
        b = Math.round(3.1 * perc);
    } else {
        g = Math.round(510 - 5.1 * perc);
        r = 255;
    }
    var h = r * 0x10000 + g * 0x100 + b * 0x1;
    return '#' + ('000000' + h.toString(16)).slice(-6);
}

function downloadCanvas() {  //function that download the Canvas in png
    var link = document.createElement('a');
    link.download = "canvas.png";
    link.href = document.getElementById("myCanvas").toDataURL("image/png");
    link.click();
}
