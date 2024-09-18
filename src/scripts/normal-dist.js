let my={}
http:function init(){let version='0.78';let w=720;let h=410;my.wd=w
my.ht=h
my.mid=w/2;this.sdMax=3.6;this.pxPer=w/(this.sdMax*2);this.perPx=(this.sdMax*2)/w;this.yZero=300;my.animQ=true;my.currZVal=0;
var typs=[['0z',"0 to Z"],['toz',"Up to Z"],['zon',"Z onwards"]];
let s=wrap({cls:'js',style:'width:'+w+'px; min-height:'+h+'px; background-color: var(--b1Clr);'},wrap({id:'canvasId',tag:'can'}),
wrap({tag:'rad',fn:'go('+(w/2-100)+')',opts:typs,pos:'abs',style:'left:5px; top:3px'}),
wrap({id:'info',cls:'output',pos:'abs',style:'font: 25px Arial; right:5px; top:5px; height:30px; min-width:230px;'}),
wrap({pos:'abs',style:'font-size: 0.9rem; right:5px; top:45px; height:30px; text-align:right; z-index: 100;'},
'Note: Click to Freeze/Unfreeze<br>Left/right to adjust'),wrap({cls:'copyrt',pos:'abs',style:'left:5px; bottom:3px'},
` &copy; 2021 MathsIsFun.com v${version}`))
docInsert(s);let div=document.getElementById('info')
let clr=getComputedStyle(div).getPropertyValue("color");console.log('clr',clr)
console.log('clr2',colorFromCSSClass('large'))
my.can=new Can('canvasId',w,h,2)
let el=my.can.el
el.addEventListener("mousemove",onmouseMove,false);el.addEventListener("mousedown",onmouseDown,false);window.addEventListener('touchmove',ontouchmove,false);
 window.addEventListener("keydown",onKeyPress,false);go(my.mid+50);}
function colorFromCSSClass(className){let tmp=document.createElement("div")
tmp.className=className;document.body.appendChild(tmp);let clr=getComputedStyle(tmp).getPropertyValue("background-color");document.body.removeChild(tmp);return clr}
function go(x){let g=my.can.g
g.clearRect(0,0,g.canvas.width,g.canvas.height)
drawDist('edge');drawDist(getType(),x);drawGrid('white');drawBubble(my.midX,0,150,x);drawzScore(getType(),x);}
function getZScore(zVal){var zScore=0;
    switch(getType()){
    case "0z":zScore=Math.abs(normalcdf(zVal)-0.5);if(zVal==0)zScore=0;break;
    case "toz":zScore=normalcdf(zVal);if(zVal==0)zScore=0.5;break;
    case "zon":zScore=1-normalcdf(zVal);if(zVal==0)zScore=0.5;break;default:}
return(Math.round(zScore*10000)/100).toString()+"%";}
function normalcdf(X){var T=1/(1+0.2316419*Math.abs(X));var D=0.3989423*Math.exp(-X*X/2);var Prob=D*T*(0.3193815+T*(-0.3565638+T*(1.781478+T*(-1.821256+T*1.330274))));if(X>0){Prob=1-Prob}
return Prob}
function getType(){var typeStr='';var buttons=document.getElementsByName('typ');for(var i=0;i<buttons.length;i++){var button=buttons[i];if(button.checked){typeStr=button.value;}}
return typeStr;}
function drawDist(typ,currX){let g=my.can.g
   g.beginPath();g.strokeStyle='blue';g.fillStyle='rgba(0,0,255,0.5)';var sttQ=true;
  var f=(1/Math.sqrt(2*Math.PI));currX=Math.max(currX,20);currX=Math.min(currX,my.wd-20);var stti=0;var endi=my.wd
switch(typ){case 'edge':g.moveTo(0,this.yZero);g.lineTo(my.wd,this.yZero);g.stroke();g.beginPath();break;
    case '0z':stti=my.wd/2;endi=currX;break;
    case 'toz':stti=0;endi=currX;break;
    case 'zon':stti=currX;endi=my.wd;break;default:}
if(stti>endi){let temp=endi;endi=stti;stti=temp;}
var zVal,y;for(var i=stti;i<=endi;i+=1){zVal=toXVal(i);y=f*Math.exp(-0.5*zVal*zVal);var yp=this.yZero-y*700;if(sttQ){g.moveTo(i,yp);sttQ=false;}
    else{g.lineTo(i,yp);}}
switch(typ){case 'edge':g.stroke();break;
    case '0z':case 'toz':case 'zon':g.lineTo(endi,this.yZero);g.lineTo(stti,this.yZero);g.closePath();g.fill();break;default:}}
function drawzScore(typ,currX){if(typ=='edge')return;var zVal=toXVal(currX);zVal=zRound(zVal);var zScoreStr=getZScore(zVal);
var xp=currX;
var descr='';
    switch(typ){
      case '0z':xp=(currX+my.mid)/2;xp=Math.max(260,Math.min(xp,my.wd-260));
         if(zVal<0){descr=zVal.toString()+' to 0';}
         else{descr='0 to '+ zVal.toString();}
              break;
        case 'toz':xp=(currX+200)/2;xp=Math.max(260,Math.min(xp,my.mid));descr='Up to '+zVal.toString();break;
        case 'zon':xp=(currX+my.wd-200)/2;xp=Math.max(my.mid,Math.min(xp,my.wd-260));descr=zVal.toString()+' onwards';break;default:}
let g=my.can.g
g.fillStyle=my.can.txtClr
g.font='bold 40px Arial';g.fillText(zScoreStr,xp,this.yZero-50);var s='';s+=descr;s+=': ';s+=zScoreStr;document.getElementById('info').innerHTML=s;}
function drawGrid(){let g=my.can.g
    g.textAlign='center';for(var i=-3;i<=3;i++){var xp=toXPix(i);g.beginPath();g.strokeStyle=my.can.bgClr
    g.moveTo(xp,0);g.lineTo(xp,my.ht);g.stroke();g.fillStyle='blue';g.font='bold 22px Arial';g.fillText(i.toString(),xp,this.yZero+20);}
    g.font='bold 20px Arial';g.fillText('Standard Deviations',my.mid,this.yZero+45);}
function drawBubble(x,y,wd,currX){let g=my.can.g
    g.textAlign='center';g.lineWidth=1;currX=Math.max(currX,0);currX=Math.min(currX,my.wd);g.font='bold 17px Arial';g.fillStyle='orange';g.strokeStyle=g.fillStyle;var zVal=toXVal(currX);zVal=zRound(zVal);var absTxt='Z = '+zVal.toString();g.drawSpeechBubble(absTxt,currX,this.yZero,wd,50,-25,my.wd+25,55);}
function toXPix(val){return(val+this.sdMax)*this.pxPer;}
function toXVal(pix){return(pix-my.mid)*this.perPx;}
function ontouchstart(evt){var touch=evt.targetTouches[0];evt.clientX=touch.clientX;evt.clientY=touch.clientY;evt.touchQ=true;onmouseDown(evt)}
function ontouchmove(evt){var touch=evt.targetTouches[0];evt.clientX=touch.clientX;evt.clientY=touch.clientY;evt.touchQ=true;onmouseMove(evt);
    evt.preventDefault();}
function ontouchend(ev){let el=my.can.el
    el.addEventListener('touchstart',ontouchstart,false);window.removeEventListener("touchend",ontouchend,false);}
function onmouseDown(ev){let[mouseX,mouseY]=my.can.mousePos(ev)
    my.animQ=!my.animQ;if(my.animQ)go(mouseX);ev.preventDefault();}
function onmouseMove(ev){if(my.animQ){let[mouseX,mouseY]=my.can.mousePos(ev)
    var zVal=toXVal(mouseX);my.currZVal=zRound(zVal);go(mouseX);}}
function onKeyPress(ev){var keyCode=ev.keyCode;if(keyCode==37){my.currZVal=Math.round(my.currZVal*100-1)/100;var currX=toXPix(my.currZVal);go(currX);ev.preventDefault();}
    if(keyCode==39){my.currZVal=Math.round(my.currZVal*100+1)/100;let currX=toXPix(my.currZVal);go(currX);ev.preventDefault();}}
function Point(x,y){this.x=x;this.y=y;}
    Point.prototype.set=function(x,y){this.x=x;this.y=y;};CanvasRenderingContext2D.prototype.drawArrow=function(x0,y0,totLen,shaftHt,headLen,headHt,angle,sweep,invertQ){var g=this;var pts=[[0,0],[-headLen,-headHt/2],[-headLen+sweep,-shaftHt/2],[-totLen,-shaftHt/2],[-totLen,shaftHt/2],[-headLen+sweep,shaftHt/2],[-headLen,headHt/2],[0,0]];if(invertQ){pts.push([0,-headHt/2],[-totLen,-headHt/2],[-totLen,headHt/2],[0,headHt/2]);}
    for(var i=0;i<pts.length;i++){var cosa=Math.cos(-angle);var sina=Math.sin(-angle);var xPos=pts[i][0]*cosa+pts[i][1]*sina;var yPos=pts[i][0]*sina-pts[i][1]*cosa;if(i==0){g.moveTo(x0+xPos,y0+yPos);}else{g.lineTo(x0+xPos,y0+yPos);}}};CanvasRenderingContext2D.prototype.drawSpeechBubble=function(txt,xp,yp,wd,ht,xMin,xMax,ptrHt){var g=this;g.fillStyle="rgba(230, 200, 50, 0.7)";g.beginPath();g.moveTo(xp,yp);g.lineTo(xp+ptrHt/3,yp+ptrHt-5);g.lineTo(xp-ptrHt/3,yp+ptrHt-5);g.lineTo(xp,yp);var boxX=Math.max(xMin,xp-wd/2);boxX=Math.min(xMax-wd,boxX);g.roundRect(boxX,yp+ptrHt-5,wd,ht,30);g.closePath();g.fill();g.fillStyle="rgba(0, 0, 50, 0.7)";g.font="bold 28px Arial";g.textAlign='center';g.fillText(txt,boxX+wd/2,yp+ht*1.7,wd);};CanvasRenderingContext2D.prototype.roundRect=function(x,y,w,h,r){if(w<2*r)r=w/2;if(h<2*r)r=h/2;this.moveTo(x+r,y);this.arcTo(x+w,y,x+w,y+h,r);this.arcTo(x+w,y+h,x,y+h,r);this.arcTo(x,y+h,x,y,r);this.arcTo(x,y,x+w,y,r);return this;};function zRound(z){return(Math.round(z*100))/100;}
function docInsert(s){let div=document.createElement('div')
    div.innerHTML=s
    let script=document.currentScript
    script.parentElement.insertBefore(div,script);}
class Can{constructor(id,wd,ht,ratio){this.id=id
    this.wd=wd
    this.ht=ht
    this.ratio=ratio
    let el=document.getElementById(id);el.width=wd*ratio;el.style.width=wd+"px";el.height=ht*ratio;el.style.height=ht+"px";
    this.g=el.getContext("2d");this.g.setTransform(ratio,0,0,ratio,0,0);this.el=el
    this.styleSet()
    return this}
styleSet(){let div=document.createElement("div")
div.className='jsCanvas'
document.body.appendChild(div);let styles=getComputedStyle(div)
this.bgClr=styles.getPropertyValue("background-color");this.txtClr=styles.getPropertyValue("color");document.body.removeChild(div);}
mousePos(ev){let bRect=this.el.getBoundingClientRect();let mouseX=(ev.clientX-bRect.left)*(this.el.width/this.ratio/bRect.width);
   let mouseY=(ev.clientY-bRect.top)*(this.el.height/this.ratio/bRect.height);return[mouseX,mouseY]}}
  function wrap({id='',cls='',pos='rel',style='',txt='',tag='div',lbl='',fn='onChg()',opts=[]},...mores){let s=''
    s+='\n'
    txt+=mores.join('')
    if(tag=='btn')s+='<button class="btn" onclick="'+fn+'"'
    if(tag=='can')s+='<canvas'
    if(tag=='div')s+='<div'
    if(tag=='inp'){if(cls.length==0)cls='input'
    if(lbl.length>0){s+='<div style="display:inline-block;">'
    s+='<span class="label">'+lbl+' </span>'}
    s+='<input value="'+txt+'"  oninput="'+fn+'" onchange="'+fn+'"'}
    if(tag=='rad'){if(cls.length==0)cls="radio"
    s+='<form';if(fn.length>0)s+=' onclick="'+fn+'"'}
    if(tag=='sel'){if(cls.length==0)cls="select"
    s+='<select onclick="'+fn+'"'}
    if(tag=='sld')s+='<input type="range" '+txt+' oninput="'+fn+'" onchange="'+fn+'"'
    if(id.length>0)s+=' id="'+id+'"'
    if(cls.length>0)s+=' class="'+cls+'"'
    if(pos=='dib')s+=' style="position:relative; display:inline-block;'+style+'"'
    if(pos=='rel')s+=' style="position:relative; '+style+'"'
    if(pos=='abs')s+=' style="position:absolute; '+style+'"'
    if(tag=='btn')s+='>'+txt+'</button>'
    if(tag=='can')s+='></canvas>'
    if(tag=='div')s+=' >'+txt+'</div>'
    if(tag=='inp'){s+='>'
    if(lbl.length>0){s+='</div>'}}
    if(tag=='rad'){s+='>\n'
    for(let i=0;i<opts.length;i++)
    {var chk='';if(i==1)chk='checked';s+='<input type="radio" id="r'+i+'" name="typ" style="cursor:pointer;" value="'+opts[i][0]+'" '+chk+' />\n';
            s+='<label for="r'+i+'" style="cursor:pointer;">'+opts[i][1]+'</label><br/>\n';}
    s+='</form>';}
    if(tag=='sel'){s+='>\n'
    for(let i=0;i<opts.length;i++){let idStr=id+i;let chkStr=i==99?' checked ':'';s+='<option id="'+idStr+'" value="'+opts[i]+'"'+chkStr+'>'+opts[i]+'</option>\n';}
    s+='</select>';}
    if(tag=='sld')s+='>'
    s+='\n'
    return s}
init()