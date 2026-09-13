import * as T from 'three';
import type {Event,Run,Presentation} from './replay';

// Original procedural artwork, inspired by rendered fly-body demos. This is
// deterministic keyframe animation, not NeuroMechFly or a motor-circuit model.
// The page marker follows recorded glyph boxes; locomotion never enters OCR.
export class ReadingFly {
  readonly renderer:T.WebGLRenderer;
  readonly scene=new T.Scene();
  readonly camera=new T.PerspectiveCamera(37,776/640,.1,100);
  readonly root=new T.Group();
  readonly head=new T.Group();
  readonly wings:T.Group[]=[];
  readonly legs:{side:number;index:number;bones:T.Mesh[];joints:T.Mesh[]}[]=[];
  readonly page:T.Mesh;
  readonly marker:T.Mesh;
  readonly eyeMaterial:T.MeshStandardMaterial;
  readonly pageTextures=new Map<string,T.CanvasTexture>();
  readonly target=new T.Vector3();
  readonly neuronGlow:T.Mesh;
  private sphere=new T.SphereGeometry(1,24,16);
  private cylinder=new T.CylinderGeometry(1,1,1,8);
  private up=new T.Vector3(0,1,0);
  private paperWidth=7.6;
  private paperHeight=9.8;
  constructor(width=776,height=640){
    this.renderer=new T.WebGLRenderer({antialias:true,alpha:false,preserveDrawingBuffer:true});
    this.renderer.setSize(width,height);this.renderer.setPixelRatio(1);
    this.renderer.setClearColor('#102029');this.renderer.outputColorSpace=T.SRGBColorSpace;
    this.renderer.toneMapping=T.ACESFilmicToneMapping;this.renderer.toneMappingExposure=1.25;
    this.renderer.shadowMap.enabled=true;this.renderer.shadowMap.type=T.PCFSoftShadowMap;
    this.camera.aspect=width/height;this.camera.updateProjectionMatrix();
    this.scene.fog=new T.Fog('#102029',19,38);
    this.scene.add(new T.HemisphereLight('#d1eeff','#4d3622',2.2));
    const key=new T.DirectionalLight('#ffe0aa',3.8);key.position.set(-4,10,-5);key.castShadow=true;
    key.shadow.mapSize.set(1024,1024);key.shadow.camera.left=-8;key.shadow.camera.right=8;
    key.shadow.camera.top=8;key.shadow.camera.bottom=-8;key.shadow.normalBias=.035;
    this.scene.add(key);
    const rim=new T.DirectionalLight('#67dcff',2.8);rim.position.set(5,4,4);this.scene.add(rim);
    const desk=new T.Mesh(new T.PlaneGeometry(70,70),new T.MeshStandardMaterial({color:'#12252c',roughness:.92}));
    desk.rotation.x=-Math.PI/2;desk.position.y=-.07;desk.receiveShadow=true;this.scene.add(desk);
    const grid=new T.GridHelper(50,50,'#22434c','#1a313a');grid.position.y=-.06;this.scene.add(grid);
    this.page=new T.Mesh(new T.PlaneGeometry(this.paperWidth,this.paperHeight),new T.MeshStandardMaterial({color:'white',roughness:.95}));
    this.page.rotation.x=-Math.PI/2;this.page.receiveShadow=true;this.scene.add(this.page);
    // A few paper edges give the PDF a physical presence on the desk.
    for(let i=1;i<=3;i++){
      const sheet=new T.Mesh(new T.BoxGeometry(this.paperWidth,.018,this.paperHeight),new T.MeshStandardMaterial({color:'#cdd3cf',roughness:1}));
      sheet.position.set(i*.018,-i*.016,i*.018);this.scene.add(sheet);
    }
    this.marker=new T.Mesh(new T.PlaneGeometry(1,1),new T.MeshBasicMaterial({color:'#ff6c3e',transparent:true,opacity:.46,depthWrite:false}));
    this.marker.rotation.x=-Math.PI/2;this.marker.position.y=.025;this.scene.add(this.marker);
    this.scene.add(this.root);
    const shell=new T.MeshStandardMaterial({color:'#704925',roughness:.53,metalness:.12});
    const gold=new T.MeshStandardMaterial({color:'#b47839',roughness:.58,metalness:.15});
    const dark=new T.MeshStandardMaterial({color:'#271c17',roughness:.62});
    const legMat=new T.MeshStandardMaterial({color:'#69472a',roughness:.55,metalness:.1});
    this.ellipsoid(this.root,shell,[0,1.04,.03],[.56,.58,.71]);
    this.ellipsoid(this.root,gold,[0,1.09,.96],[.57,.47,.94]);
    // Dark curved tergite stripes follow the abdomen's surface.
    for(let i=0;i<6;i++){
      const z=.42+i*.245,k=Math.sqrt(Math.max(.05,1-((z-.96)/.96)**2));
      const ring=new T.Mesh(new T.TorusGeometry(1,.054,6,40),dark);
      ring.position.set(0,1.09,z);ring.scale.set(.575*k,.478*k,.7);this.root.add(ring);
    }
    this.ellipsoid(this.root,dark,[0,1.08,1.82],[.2,.24,.28]);
    this.root.add(this.head);this.head.position.set(0,1.22,-.78);
    this.ellipsoid(this.head,gold,[0,0,0],[.56,.47,.4]);
    const texCanvas=document.createElement('canvas');texCanvas.width=512;texCanvas.height=256;
    const eyeCtx=texCanvas.getContext('2d')!;eyeCtx.fillStyle='#bb291c';eyeCtx.fillRect(0,0,512,256);
    for(let row=0;row<22;row++)for(let col=0;col<40;col++){
      const x=col*14+(row%2)*7,y=row*12;eyeCtx.beginPath();
      for(let k=0;k<6;k++){const a=Math.PI/3*k;eyeCtx.lineTo(x+6.8*Math.cos(a),y+6.8*Math.sin(a));}
      eyeCtx.closePath();eyeCtx.fillStyle=`rgb(${155+(col*17+row*13)%50},${28+(row*7)%20},20)`;eyeCtx.fill();
      eyeCtx.strokeStyle='#721b19';eyeCtx.lineWidth=.9;eyeCtx.stroke();
    }
    const eyeTex=new T.CanvasTexture(texCanvas);eyeTex.colorSpace=T.SRGBColorSpace;
    this.eyeMaterial=new T.MeshStandardMaterial({map:eyeTex,color:'#ff9b78',roughness:.38,metalness:.2});
    for(const side of [-1,1]){
      const eye=this.ellipsoid(this.head,this.eyeMaterial,[side*.46,.05,-.035],[.34,.5,.37]);eye.rotation.z=-side*.12;
      this.ellipsoid(this.head,dark,[side*.17,.04,-.395],[.105,.16,.1]);
      this.segment(this.head,new T.Vector3(side*.17,.09,-.45),new T.Vector3(side*.33,.46,-.63),.015,dark);
      for(let k=0;k<4;k++)this.segment(this.head,new T.Vector3(side*(.22+k*.024),.18+k*.067,-.5-k*.027),new T.Vector3(side*(.41+k*.024),.2+k*.07,-.56-k*.027),.007,dark);
      this.ellipsoid(this.head,gold,[side*.14,-.29,-.36],[.11,.1,.19]);
      // A pair of small halteres beneath the wings.
      this.segment(this.root,new T.Vector3(side*.45,1.1,.6),new T.Vector3(side*.85,1.04,.8),.025,gold);
      this.ellipsoid(this.root,gold,[side*.88,1.04,.8],[.09,.09,.13]);
      const wing=new T.Group();wing.position.set(side*.37,1.44,.1);this.root.add(wing);this.wings.push(wing);
      const shape=new T.Shape();shape.moveTo(0,0);shape.bezierCurveTo(.65,-.15,1.3,.45,1.52,1.75);
      shape.bezierCurveTo(1.51,2.51,.7,2.66,.22,1.57);shape.bezierCurveTo(-.01,1.01,-.1,.33,0,0);
      const wingGeo=new T.ShapeGeometry(shape,32);wingGeo.rotateX(Math.PI/2);wingGeo.scale(side,1,1);
      const wingMesh=new T.Mesh(wingGeo,new T.MeshStandardMaterial({color:'#c6eff2',transparent:true,opacity:.4,side:T.DoubleSide,roughness:.18,metalness:.26,depthWrite:false}));wing.add(wingMesh);
      const lineMat=new T.LineBasicMaterial({color:'#7e9a9e',transparent:true,opacity:.66});
      const paths=[[[0,0],[.42,.28],[1.03,1.2],[1.35,2.0]],[[0,0],[.25,.8],[.57,1.9],[.9,2.25]],[[.25,.8],[.85,.8],[1.34,1.5]],[[.57,1.9],[1.1,1.65]],[[.42,.28],[.6,1.25],[1.3,2.16]]];
      for(const path of paths){const geom=new T.BufferGeometry().setFromPoints(path.map(([x,z])=>new T.Vector3(side*x,.006,z)));wing.add(new T.Line(geom,lineMat));}
      for(let index=0;index<3;index++){
        const bones=Array.from({length:3},()=>{const mesh=new T.Mesh(this.cylinder,legMat);mesh.castShadow=true;this.root.add(mesh);return mesh;});
        const joints=Array.from({length:3},()=>this.ellipsoid(this.root,gold,[0,0,0],[.055,.055,.055]));
        this.legs.push({side,index,bones,joints});
      }
    }
    // Short bristles: deterministic surface points, a single line-segment mesh.
    const bristles:number[]=[];
    for(let i=0;i<160;i++){
      const a=i*2.39996,y=.22+(i%29)/35,rad=Math.sqrt(1-y*y);
      const nx=Math.cos(a)*rad,nz=Math.sin(a)*rad;
      bristles.push(nx*.56,1.04+y*.58,.03+nz*.71,nx*.66,1.04+y*.71,.03+nz*.84);
    }
    const hairs=new T.BufferGeometry();hairs.setAttribute('position',new T.Float32BufferAttribute(bristles,3));
    this.root.add(new T.LineSegments(hairs,new T.LineBasicMaterial({color:'#302619'})));
    // A subtle thorax accent is ornamental, not a map of neural locations.
    this.neuronGlow=this.ellipsoid(this.root,new T.MeshStandardMaterial({color:'#e4b35c',emissive:'#bc6a21',emissiveIntensity:.12,roughness:.4}),[0,1.575,.06],[.12,.035,.36]);
  }
  private ellipsoid(parent:T.Object3D,material:T.Material,position:number[],scale:number[]){
    const mesh=new T.Mesh(this.sphere,material);mesh.position.set(...position as [number,number,number]);mesh.scale.set(...scale as [number,number,number]);mesh.castShadow=true;parent.add(mesh);return mesh;
  }
  private bone(mesh:T.Mesh,a:T.Vector3,b:T.Vector3,radius:number){
    mesh.position.copy(a).add(b).multiplyScalar(.5);mesh.scale.set(radius,a.distanceTo(b),radius);
    mesh.quaternion.setFromUnitVectors(this.up,b.clone().sub(a).normalize());
  }
  private segment(parent:T.Object3D,a:T.Vector3,b:T.Vector3,radius:number,material:T.Material){
    const mesh=new T.Mesh(this.cylinder,material);this.bone(mesh,a,b,radius);parent.add(mesh);return mesh;
  }
  prepare(id:string,source:HTMLImageElement,crop:HTMLImageElement,presentation:Presentation){
    if(!this.pageTextures.has(id)){
      const canvas=document.createElement('canvas');canvas.width=1200;canvas.height=Math.round(1200*source.height/source.width);
      const c=canvas.getContext('2d')!;c.drawImage(source,0,0,canvas.width,canvas.height);
      const r=presentation.region;
      // Use the actual processed crop here too, including the controlled tilt.
      c.drawImage(crop,r[0]*canvas.width,r[1]*canvas.height,(r[2]-r[0])*canvas.width,(r[3]-r[1])*canvas.height);
      c.strokeStyle='#f47740';c.lineWidth=5;
      c.strokeRect(r[0]*canvas.width,r[1]*canvas.height,(r[2]-r[0])*canvas.width,(r[3]-r[1])*canvas.height);
      const texture=new T.CanvasTexture(canvas);texture.colorSpace=T.SRGBColorSpace;texture.anisotropy=4;
      this.pageTextures.set(id,texture);
    }
    (this.page.material as T.MeshStandardMaterial).map=this.pageTextures.get(id)!;
    (this.page.material as T.MeshStandardMaterial).needsUpdate=true;
  }
  render(t:number,run:Run,e:Event,next:Event,fraction:number,presentation:Presentation,crop:HTMLImageElement,finished:boolean){
    const region=presentation.region;
    const position=(event:Event)=>{
      const [x0,y0,x1,y1]=event.box;
      return new T.Vector3((region[0]+(x0+x1)/2/crop.width*(region[2]-region[0])-.5)*this.paperWidth,0,
        (region[1]+(y0+y1)/2/crop.height*(region[3]-region[1])-.5)*this.paperHeight);
    };
    const p=position(e),q=position(next);
    const smooth=fraction*fraction*(3-2*fraction);this.target.copy(p).lerp(q,smooth);
    this.marker.position.set(p.x,.025,p.z);this.marker.scale.set(Math.max(.12,(e.box[2]-e.box[0])/crop.width*(region[2]-region[0])*this.paperWidth),Math.max(.12,(e.box[3]-e.box[1])/crop.height*(region[3]-region[1])*this.paperHeight),1);
    // Body follows the scan with an offset so the head leans over its marker.
    this.root.position.set(this.target.x,.04+Math.sin(t*5)*.015,this.target.z+1.05);
    this.root.rotation.set(.04*Math.sin(t*3),-.1+.09*Math.sin(t*2),.018*Math.sin(t*5));
    this.head.rotation.set(.12+.07*Math.sin(t*6),.1*Math.sin(t*3.4),.04*Math.sin(t*4));
    this.wings.forEach((wing,i)=>{const side=i===0?-1:1;wing.rotation.z=side*(.08+.04*Math.sin(t*9));wing.rotation.y=side*(-.12+.06*Math.sin(t*3));});
    const gait=finished?.13:1;
    for(const leg of this.legs){
      const {side,index}=leg,phase=t*8+index*Math.PI+(side===1?Math.PI:0),step=Math.sin(phase)*.23*gait,lift=Math.max(0,Math.cos(phase))*.22*gait;
      const hip=new T.Vector3(side*.4,.96,(index-1)*.5);
      const knee=new T.Vector3(side*(index===1?1.16:1.04),.6,(index-1)*.98+step*.3);
      const ankle=new T.Vector3(side*(index===1?1.45:1.3),.12+lift,(index-1)*1.25+step);
      const foot=new T.Vector3(side*(index===1?1.61:1.5),.035+lift,(index-1)*1.45+step);
      [hip,knee,ankle].forEach((point,i)=>leg.joints[i].position.copy(point));
      this.bone(leg.bones[0],hip,knee,.065);this.bone(leg.bones[1],knee,ankle,.04);this.bone(leg.bones[2],ankle,foot,.019);
    }
    this.camera.position.set(this.target.x*.55+6.6,8.7,this.target.z*.55-8.9);
    this.camera.lookAt(this.target.x*.65,.45,this.target.z*.6+.2);
    this.renderer.render(this.scene,this.camera);
    return this.renderer.domElement;
  }
}
