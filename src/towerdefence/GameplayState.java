/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package towerdefence;

import java.util.ArrayList;
import java.awt.Font;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.newdawn.slick.Animation;
import org.newdawn.slick.Color;
import org.newdawn.slick.TrueTypeFont;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.Image;
import org.newdawn.slick.Input;
import org.newdawn.slick.MouseListener;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.SpriteSheet;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.particles.ParticleSystem;
import org.newdawn.slick.state.BasicGameState;
import org.newdawn.slick.state.StateBasedGame;
import org.newdawn.slick.tiled.TiledMap;
import org.newdawn.slick.util.pathfinding.*;
import towerdefence.engine.AnimationLoader;
import towerdefence.engine.Settings;
import towerdefence.engine.Wallet;
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
    private PathMap pathmap;
    // Path finder used to search the pathmap
    private PathFinder finder;
    // Gives the last path found for the current unit
    private TiledMap map;
    // Map for gui underlay
    private TiledMap guiMap;

    CritterManager critterManager;
    private int waveNumber;
//    ArrayList<CritterManager> critterWaveList = new ArrayList<CritterManager>();
//    CritterManager critterWave;
    ArrayList<Critter> critterList;
    ArrayList<Tower> towerList = new ArrayList<Tower>();

    Critter critter = null;
    ParticleSystem ps;

    private TrueTypeFont trueTypeFont;
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

    GameplayState(int stateID) {
        this.stateID = stateID;
    }

    @Override
    public int getID() {
        return 1;
    }

    public void init(GameContainer container, StateBasedGame game) throws SlickException {
        
        settings = new Settings();
        
        
        
        // Load Level
        level = new LevelLoader("data/levels/snake.xml");
        guiMap = new TiledMap("data/gui/guiMap.tmx");

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
        
        towerFactory = new TowerManager(towerSprites);
        
        critterManager = new CritterManager(startX, startY,
                targetX, targetY, finder,critterAnimation);
        waveNumber = 0;
        critterCount = level.getWave(waveNumber).getNumCritters();
        
        waterAnimation = new renderWater(map.getWidth()+5,map.getHeight());

        Font font = new Font("Verdana", Font.PLAIN, 20);
        trueTypeFont = new TrueTypeFont(font, true);
        
        startWaves = false;

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
        Wallet.getInstance().setCash(startingMoney);
        Wallet.getInstance().setCritterReward(critterReward);
        Wallet.getInstance().setTowerCost(towerCost);
        
    }

    public void render(GameContainer container, StateBasedGame game, Graphics g) throws SlickException {
        g.setClip(TILESIZE, TILESIZE, TILESIZE*(map.getWidth()+3) , TILESIZE*(map.getHeight()-2));
        
        waterAnimation.render(container, game, g);
        map.render(0, 0,1);
        map.render(0, 0,2);
        guiMap.render(21*32, 0);


//        for(CritterManager wave : critterWaveList) {
//            wave.render(container, game, g);
//            tempCritterCount+=wave.getCritters().size();
//        }

        critterManager.render(container, game, g);
        tempCritterCount=critterManager.getCritters().size();

        towerFactory.render(container, game, g);
        
        trueTypeFont.drawString(50, 200,
                "Cash : $" + String.valueOf(Wallet.getInstance().getCash()), Color.white);
        
        trueTypeFont.drawString(50, 110,
                "# of Critters : " + String.valueOf(tempCritterCount), Color.white);
        trueTypeFont.drawString(50, 160,
                "# of Towers : " + String.valueOf(towerFactory.getTowers().size()), Color.white);
        if(startWaves && waveCounter>0) {
            trueTypeFont.drawString(50, 530,
                "Next wave in : " + String.valueOf(waveCounter/1000.0) + " seconds", Color.white);
        }

        if(!startWaves) {
            trueTypeFont.drawString(50, 530,
                "Press Enter to begin waves", Color.white);
        }

        wanderingNPCAnim.draw(50f, 32f);

        if(selectedTower!=null) {
            selectedTower.render(container, game, g);
        }

    }

    public void update(GameContainer container, StateBasedGame game, int delta) throws SlickException {

        mouseCounter -= delta;
        waveCounter -= delta;
        generateCounter -= delta;

//        ArrayList<Critter> tempCritterList = new ArrayList<Critter>();

        Input input = container.getInput();

        if (mouseCounter <= 0) {
            
            if (input.isKeyPressed(Input.KEY_F2)) {
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
                setSelectedTower(input,TowerManager.NORMAL);
            }
            if (input.isKeyPressed(Input.KEY_2)) {
                setSelectedTower(input,TowerManager.FIRE);
            }
            if (input.isKeyPressed(Input.KEY_3)) {
                setSelectedTower(input,TowerManager.ICE);;
            }
        }


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

        input.addMouseListener(new MouseListener(){

            public void mouseWheelMoved(int change) {
            }

            public void mouseClicked(int button, int x, int y, int clickCount) {
                if (mouseCounter <= 0) {
                    int currentXTile = (int) Math.floor((x / GameplayState.TILESIZE));
                    int currentYTile = (int) Math.floor((y / GameplayState.TILESIZE));
                    if (button == 0) {
                        if(currentXTile <= 21) {
                            if (pathmap.getTerrain(currentXTile, currentYTile) != PathMap.GRASS
                                    && pathmap.getTerrain(currentXTile, currentYTile) != PathMap.NOPLACE) {
                                try {
                                    java.util.Random towerType = new Random();
                                    towerFactory.addTower(String.valueOf(x),
                                            new Vector2f(currentXTile * GameplayState.TILESIZE,
                                            currentYTile * GameplayState.TILESIZE),towerType.nextInt(3));

                                    pathmap.setTowerTerrain(new Vector2f(currentXTile, currentYTile));

                                    mouseCounter = 100;
                                } catch (SlickException ex) {
                                    Logger.getLogger(GameplayState.class.getName()).log(Level.SEVERE, null, ex);
                                }
                            }
                        }
                    }
//                    } else if (button == 1) {
//                        if (pathmap.getTerrain(currentXTile, currentYTile) == PathMap.GRASS) {
//                            try {
//                                CritterManager newWave = new CritterManager(
//                                        new Vector2f(currentXTile * GameplayState.TILESIZE,
//                                            currentYTile * GameplayState.TILESIZE),
//                                            targetX,targetY,finder, critterCount, CritterManager.NORMAL);
//
//                                critterWaveList.add(newWave);
//                                mouseCounter = 100;
//                            } catch (SlickException ex) {
//                                Logger.getLogger(GameplayState.class.getName()).log(Level.SEVERE, null, ex);
//                            }
//                        }
//                    }
                }
            }

            public void mousePressed(int button, int x, int y) {
                if(button==1) {
                    selectedTower=null;
                }
            }

            public void mouseReleased(int button, int x, int y) {
            }

            public void mouseMoved(int oldx, int oldy, int newx, int newy) {
                if(selectedTower!=null) {
                    selectedTower.setPosition(new Vector2f(newx-16,newy-16));
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
        
        if(selectedTower!=null) {
            selectedTower.update(container, game, delta);
        }

    }

    private void setSelectedTower(Input input, int type) throws SlickException {
        selectedTower = new Tower("selected");
        selectedTower.setType(type);
        selectedTower.AddComponent(new ImageRenderComponent("CritterRender",
                towerSprites[type][0]));
        selectedTower.setPosition(new Vector2f(input.getMouseX()-16,
                input.getMouseY()-16));
    }

}
