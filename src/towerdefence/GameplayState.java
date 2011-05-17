/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package towerdefence;

import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.newdawn.slick.Animation;
import org.newdawn.slick.Color;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.Image;
import org.newdawn.slick.Input;
import org.newdawn.slick.MouseListener;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.SpriteSheet;
import org.newdawn.slick.UnicodeFont;
import org.newdawn.slick.font.effects.ColorEffect;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.particles.ParticleSystem;
import org.newdawn.slick.state.BasicGameState;
import org.newdawn.slick.state.StateBasedGame;
import org.newdawn.slick.tiled.TiledMap;
import org.newdawn.slick.util.pathfinding.*;
import towerdefence.engine.AnimationLoader;
import towerdefence.engine.Settings;
import towerdefence.engine.Player;
import towerdefence.engine.component.ImageRenderComponent;
import towerdefence.engine.entity.Critter;
import towerdefence.engine.entity.Tower;
import towerdefence.engine.levelLoader.LevelLoader;
import towerdefence.engine.levelLoader.Wave;

import towerdefence.engine.pathfinding.PathMap;

/**
 *
 * @author Jiv
 */
public class GameplayState extends BasicGameState {

    int stateID = -1;

    // Tower and Critter types
    public final static int NORMAL = 0;
    public final static int FIRE = 1;
    public final static int ICE = 2;
    // Critter type
    public final static int BOSS = 3;
    
    public Settings settings;
    
    private AnimationLoader spriteLoader = new AnimationLoader();
    private Image[][] towerSprites  = new Image[3][];
    private Animation[][] critterAnimation = new Animation[3][];
    private Image tileHighlight;
    private Image validTile;
    private Image invalidTile;
    
    Tower selectedTower=null;
    

    LevelLoader level;
    
    int critterCount;
    int tempCritterCount;

    // Update Time Counters
    int mouseCounter;
    int waveCounter;
    int generateCounter;

    boolean startWaves;

    // Map used to find critter paths
    public static PathMap pathmap;
    // Path finder used to search the pathmap
    private PathFinder finder;
    // Gives the last path found for the current unit
    private TiledMap map;
    
    
    // Map for gui underlay
    private Image guiBackground;
    private int guiLeftX = (int)((21*32) + 16);
    private int guiTopY = 48;
    private int guiBottomY = 656;

    CritterManager critterManager;
    private int waveNumber;
//    ArrayList<CritterManager> critterWaveList = new ArrayList<CritterManager>();
//    CritterManager critterWave;
    ArrayList<Critter> critterList;
    ArrayList<Tower> towerList = new ArrayList<Tower>();
    ArrayList<Tower> guiTowerList = new ArrayList<Tower>();

    Critter critter = null;
    ParticleSystem ps;

    private UnicodeFont unicodeFont;
    private TowerManager towerFactory;
    private renderWater waterAnimation;
    private Animation wanderingNPCAnim;

    public static int TILESIZE;
    private int startX;
    private int startY;
    private int targetX;
    private int targetY;
    
    public static int startingMoney;
    public static int playerHealth;
    public static int[] critterHealth;
    public static double[] critterSpeed;
    public static int[] baseDPS;
    public static int[] towerRange;
    public static boolean[] lockOn;
    public int[] critterReward;
    public int[] towerCost;
    
    private int mouseY;
    private int mouseX;
    public static boolean towerSelected;

    GameplayState(int stateID) {
        this.stateID = stateID;
    }

    @Override
    public int getID() {
        return 1;
    }

    public void init(GameContainer container, StateBasedGame game) throws SlickException {
        
        //loadLevel("data/levels/snake.xml");
        settings = new Settings();
//        guiMap = new TiledMap("data/gui/guiMap.tmx");
        guiBackground = new Image("data/gui/gui_overlay.png");

        map = new TiledMap(level.getMapPath());

        // Get Start and Target positions from map
        startX = Integer.parseInt(map.getMapProperty("startX", null));
        startY = Integer.parseInt(map.getMapProperty("startY", null));
        targetX = Integer.parseInt(map.getMapProperty("targetX", null));
        targetY = Integer.parseInt(map.getMapProperty("targetY", null));

        TILESIZE=map.getTileWidth();
        
        pathmap = new PathMap(map);
        finder = new AStarPathFinder(pathmap, 500, false);
        
        loadResources();
        
        setGuiTowers();
        
        towerFactory = new TowerManager(towerSprites);
        
        critterManager = new CritterManager(startX, startY,
                targetX, targetY, finder,critterAnimation);
        waveNumber = 0;
        critterCount = level.getWave(waveNumber).getNumCritters();
        
        waterAnimation = new renderWater(map.getWidth()+5,map.getHeight());
        
        unicodeFont = new UnicodeFont("data/fonts/Jellyka_Estrya_Handwriting.ttf", 50, false, false);
//        unicodeFont = new UnicodeFont("data/fonts/ArchitectsDaughter.ttf", 13, false, false);
        unicodeFont.getEffects().add(new ColorEffect(java.awt.Color.white));

        startWaves = false;

    }

    public void loadLevel(String levelPath) throws SlickException {
        // Load Level
        level = new LevelLoader(levelPath);
    }

    private void generateWaves() throws SlickException {
        // Reads level and generates the waves
        if(startWaves) {
            if (waveNumber < level.getNumWaves()) {
                Wave currentWave = level.getWave(waveNumber);
                if (waveCounter <= 0) {
                    if (generateCounter <= 0) {
                        if (critterCount > 0) {
                            critterManager.addCritter(String.valueOf((waveNumber * 1000) + critterCount),
                                    level.getWave(waveNumber).getCritterType());
                            critterCount--;
                            generateCounter = 1000;
                        } else if (critterCount <=0 ) {
                            waveCounter = currentWave.getTimeToWait();
                            waveNumber++;
                            if(waveNumber<level.getNumWaves()){
                                critterCount = level.getWave(waveNumber).getNumCritters();
                            }
                        }
                        
                    }
                } 
            }
        }
    }

    private void loadResources() throws SlickException {
        // Load tower sprites
        towerSprites[NORMAL] = spriteLoader.getTowerSprites(NORMAL);
        towerSprites[FIRE] = spriteLoader.getTowerSprites(FIRE);
        towerSprites[ICE] = spriteLoader.getTowerSprites(ICE);
        critterAnimation[NORMAL] = spriteLoader.getCritterAnimation(NORMAL);
        critterAnimation[FIRE] = spriteLoader.getCritterAnimation(FIRE);
        critterAnimation[ICE] = spriteLoader.getCritterAnimation(ICE);
        
        // Load the wandering NPC - He who's purpose is to question purpose
        Image normalSheet = new Image("data/sprites/wandering_trader2.png");
        SpriteSheet critterSheet = new SpriteSheet(normalSheet, 32, 64);
        Image[] wanderingNPC = {critterSheet.getSprite(0, 0),critterSheet.getSprite(1, 0),
            critterSheet.getSprite(2, 0), critterSheet.getSprite(3, 0),
            critterSheet.getSprite(4, 0), critterSheet.getSprite(5, 0)};
        wanderingNPCAnim = new Animation(wanderingNPC, 230,true);
        validTile = new Image("data/sprites/validTileSelect.png");
        invalidTile = new Image("data/sprites/invalidTileSelect.png");
        tileHighlight = validTile;
        
        // Load settings
        startingMoney = settings.getStartingMoney();
        playerHealth = settings.getPlayerHealth();
        critterHealth = settings.getCritterHealth();
        critterSpeed = settings.getCritterSpeed();
        baseDPS = settings.getBaseDPS();
        towerRange = settings.getRange();
        critterReward = settings.getReward();
        towerCost = settings.getCost();
        
        // Initialise wallet singleton
        Player.getInstance().setCash(startingMoney);
        Player.getInstance().setCritterReward(critterReward);
        Player.getInstance().setTowerCost(towerCost);
        Player.getInstance().setHealth(playerHealth);
        
    }

    private void mouseListener(Input input) {
        input.addMouseListener(new MouseListener(){

            public void mouseWheelMoved(int change) {
            }

            public void mouseClicked(int button, int x, int y, int clickCount) {
                if (mouseCounter <= 0) {
                    int currentXTile = (int) Math.floor((x / GameplayState.TILESIZE));
                    int currentYTile = (int) Math.floor((y / GameplayState.TILESIZE));
                    if (button == 0) {
                        if (currentXTile <= 21 && selectedTower != null) {
                            if (pathmap.getTerrain(currentXTile, currentYTile) != PathMap.GRASS
                                    && pathmap.getTerrain(currentXTile, currentYTile) != PathMap.NOPLACE) {
//                                try {
//                                    java.util.Random towerType = new Random();
//                                    towerFactory.addTower(String.valueOf(x),
//                                            new Vector2f(currentXTile * GameplayState.TILESIZE,
//                                            currentYTile * GameplayState.TILESIZE),towerType.nextInt(3));
                                selectedTower.setIsPlaced(true);
                                selectedTower.setIsActive(true);
                                selectedTower.setPosition(new Vector2f(currentXTile * TILESIZE, currentYTile * TILESIZE));
                                towerFactory.addTower(selectedTower);
                                selectedTower = null;
                                pathmap.setTowerTerrain(new Vector2f(currentXTile, currentYTile));

                                mouseCounter = 100;
                            }
                        } else if(currentXTile > 21) {
                            for(Tower guiTower : guiTowerList) {
                                if(currentXTile==guiTower.getTilePosition().x 
                                        && currentYTile==guiTower.getTilePosition().y) {
                                    try {
                                        setSelectedTower(guiTower.getType());
                                        mouseCounter=100;
                                    } catch (SlickException ex) {
                                        Logger.getLogger(GameplayState.class.getName()).log(Level.SEVERE, null, ex);
                                    }
                                }
                            }
                        }
                    }
                    if(button==1 && selectedTower!=null) {
                        selectedTower=null;
                    }
                }
            }

            public void mousePressed(int button, int x, int y) {
                
            }

            public void mouseReleased(int button, int x, int y) {
            }

            public void mouseMoved(int oldx, int oldy, int newx, int newy) {
                
                int currentXTile = (int) Math.floor((newx / GameplayState.TILESIZE));
                if (selectedTower != null) {
                        selectedTower.setPosition(new Vector2f(newx-16,newy-16));
                    }
                
                if (mouseCounter <= 0 && currentXTile <= 21) {
                    int currentYTile = (int) Math.floor((newy / GameplayState.TILESIZE));
                    mouseX = newx;
                    mouseY = newy;
                    
                    
                    if (pathmap.getTerrain(currentXTile, currentYTile) != PathMap.GRASS
                            && pathmap.getTerrain(currentXTile, currentYTile) != PathMap.NOPLACE) {
                        tileHighlight = validTile;
                    } else {
                        tileHighlight = invalidTile;
                    }
                    mouseCounter = 32;
                } else if (currentXTile>21) {
                    tileHighlight = null;
                }
            }

            public void mouseDragged(int oldx, int oldy, int newx, int newy) {
                
            }

            public void setInput(Input input) {
            }

            public boolean isAcceptingInput() {
                return true;
            }

            public void inputEnded() {
            }

            public void inputStarted() {
            }
        });
    }

    private void renderGuiText() {
        
        unicodeFont.drawString(guiLeftX+15, guiTopY+5,
                "Level: "+String.valueOf(level.getLevelName()));
        
//        unicodeFont.addGlyphs("~!@!#!#$%___--");
//        
        unicodeFont.drawString(guiLeftX+15, guiTopY+65,
                "Cash : $"+String.valueOf(Player.getInstance().getCash()));
        
        unicodeFont.drawString(guiLeftX+15, guiTopY+125,
                "Health : "+String.valueOf(Player.getInstance().getHealth()));
        
//        unicodeFont.drawString(50, 110,
//                "# of Critters : " + String.valueOf(tempCritterCount), Color.white);
//        unicodeFont.drawString(50, 160,
//                "# of Towers : " + String.valueOf(towerFactory.getTowers().size()), Color.white);
        
        unicodeFont.drawString(guiLeftX+25, guiTopY+200, "TOWERS");
        
        if (startWaves) {
            unicodeFont.drawString(guiLeftX + 15, guiBottomY - 100,
                    "Next wave in :", Color.white);
            if (waveCounter > 0) {
                unicodeFont.drawString(guiLeftX + 30, guiBottomY - 70,
                        String.valueOf(waveCounter / 1000) + " seconds", Color.white);
            }
        } else if (!startWaves) {
            unicodeFont.drawString(guiLeftX + 25, guiBottomY - 95,
                    "Press Enter", Color.white);
            unicodeFont.drawString(guiLeftX + 25, guiBottomY - 75,
                    "   to begin", Color.white);
        }
    }
    
    private void setSelectedTower(int type) throws SlickException {
        selectedTower = new Tower("selected", false);
        selectedTower.setPosition(new Vector2f(mouseX-16,mouseY-16));
        selectedTower.setType(type);
        selectedTower.setSprites(towerSprites[type]);
        selectedTower.AddComponent(new ImageRenderComponent("TowerRender",
                towerSprites[type][0]));
        selectedTower.setIsPlaced(false);
    }
    
    private void setGuiTowers() throws SlickException {
        guiTowerList.add(new Tower("normal", false));
        guiTowerList.add(new Tower("fire", false));
        guiTowerList.add(new Tower("ice", false));
        
        for(int i=0;i<3;i++) {
            guiTowerList.get(i).setPosition(new Vector2f( 
                    (float)(Math.floor((guiLeftX+16+(i*32)) / GameplayState.TILESIZE))*32, 
                    (float)(Math.floor((guiTopY+272)/GameplayState.TILESIZE))*32));
            guiTowerList.get(i).setType(i);
            guiTowerList.get(i).setSprites(towerSprites[i]);
            guiTowerList.get(i).AddComponent(new ImageRenderComponent("TowerRender",
                towerSprites[i][0]));
            guiTowerList.get(i).setIsPlaced(true);
        }
        
    }

    public void update(GameContainer container, StateBasedGame game, int delta) throws SlickException {

        mouseCounter -= delta;
        waveCounter -= delta;
        generateCounter -= delta;
        
        unicodeFont.loadGlyphs(1);

//        ArrayList<Critter> tempCritterList = new ArrayList<Critter>();

        Input input = container.getInput();

        if (mouseCounter <= 0) {
            
            if (input.isKeyPressed(Input.KEY_F1)) {
                loadLevel("data/levels/snake.xml");
                container.reinit();
                mouseCounter=100;
            }
            if (input.isKeyPressed(Input.KEY_F2)) {
                loadLevel("data/levels/fork.xml");
                container.reinit();
                mouseCounter=100;
            }
            if (input.isKeyPressed(Input.KEY_0)) {
                game.enterState(TowerDefence.CUDATESTSTATE);
                mouseCounter=100;
            }

            if (input.isKeyPressed(Input.KEY_ENTER)) {
                startWaves = true;
                mouseCounter=100;
            }
            
            if (input.isKeyPressed(Input.KEY_R)) {
                container.reinit();
            }
            
            if (input.isKeyPressed(Input.KEY_P)) {
                if(container.isPaused()) {
                    container.resume();
                } else {
                    container.pause();
                }
            }
            
            if (input.isKeyPressed(Input.KEY_1)) {
                setSelectedTower(TowerManager.NORMAL);
            }
            if (input.isKeyPressed(Input.KEY_2)) {
                setSelectedTower(TowerManager.FIRE);
            }
            if (input.isKeyPressed(Input.KEY_3)) {
                setSelectedTower(TowerManager.ICE);;
            }
        }
        generateWaves();

//        for(CritterManager wave : critterWaveList) {
//            wave.update(container, game, delta);
//            // Adds critters from each wave to the temp critter list
//            tempCritterList.addAll(wave.getCritters());
//        }
        critterManager.update(container, game, delta);
//        tempCritterList=critterManager.getCritters();
        if(critterManager.getCritters()!=towerFactory.getCritterList()) {
            towerFactory.updateCritterList(critterManager.getCritters());
        }
        
        towerFactory.update(container, game, delta);

        mouseListener(input);
        
        for(Tower guiTower : guiTowerList) {
            guiTower.update(container, game, delta);
        }
        
        if(selectedTower!=null) {
            towerSelected = true;
            selectedTower.update(container, game, delta);
        } else {
            towerSelected = false;
        }

    }
    
    public void render(GameContainer container, StateBasedGame game, Graphics g) throws SlickException {
        g.setClip(TILESIZE, TILESIZE, TILESIZE*(map.getWidth()+3) , TILESIZE*(map.getHeight()-2));
        
        waterAnimation.render(container, game, g);
        
        map.render(0, 0,1);
        map.render(0, 0,2);
        
        if (tileHighlight != null) {
            tileHighlight.draw(((int) Math.floor((mouseX / GameplayState.TILESIZE))) * GameplayState.TILESIZE,
                    ((int) Math.floor((mouseY / GameplayState.TILESIZE))) * GameplayState.TILESIZE);
        }
        
        guiBackground.draw(21*32, 0);


//        for(CritterManager wave : critterWaveList) {
//            wave.render(container, game, g);
//            tempCritterCount+=wave.getCritters().size();
//        }

        critterManager.render(container, game, g);
        tempCritterCount=critterManager.getCritters().size();

        towerFactory.render(container, game, g);
        renderGuiText();

        wanderingNPCAnim.draw(50f, 32f);

        for(Tower guiTower : guiTowerList) {
            guiTower.render(container, game, g);
        }
        
        if(selectedTower!=null) {
            selectedTower.render(container, game, g);
        }

    }

}
