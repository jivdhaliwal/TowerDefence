/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package towerdefence;

import java.util.ArrayList;
import java.awt.Font;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.newdawn.slick.Animation;
import org.newdawn.slick.Color;
import org.newdawn.slick.TrueTypeFont;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.Image;
import org.newdawn.slick.Input;
import org.newdawn.slick.KeyListener;
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
import towerdefence.engine.entity.Critter;
import towerdefence.engine.entity.CudaTower;
import towerdefence.engine.levelLoader.LevelLoader;
import towerdefence.engine.levelLoader.Wave;

import towerdefence.engine.pathfinding.PathMap;

/**
 *
 * @author Jiv
 */
public class CudaTestState extends BasicGameState {

    int stateID = -1;

    // Tower and Critter types
    public final static int NORMAL = 0;
    public final static int FIRE = 1;
    public final static int ICE = 2;
    // Critter type
    public final static int BOSS = 3;
    
    private AnimationLoader spriteLoader = new AnimationLoader();
    private Animation[][] critterAnimation = new Animation[3][];

    LevelLoader level;
    
    int critterCount;
    int tempCritterCount;

    // Update Time Counters
    int mouseCounter;
    int waveCounter;
    int generateCounter;

    boolean startWaves = false;

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
    ArrayList<CudaTower> towerList = new ArrayList<CudaTower>();

    Critter critter = null;
    ParticleSystem ps;

    private TrueTypeFont trueTypeFont;
    private CudaTowerManager towerFactory;
    private renderWater waterAnimation;
    private Animation wanderingNPCAnim;

    public static int TILESIZE;
    private int startX;
    private int startY;
    private int targetX;
    private int targetY;




    CudaTestState(int stateID) {
        this.stateID = stateID;
    }

    @Override
    public int getID() {
        return 3;
    }

    public void init(GameContainer container, StateBasedGame game) throws SlickException {

        level = new LevelLoader("data/levels/snake.xml");
        guiMap = new TiledMap("data/gui/guiMap.tmx");

        map = new TiledMap(level.getMapPath());

        startX = Integer.parseInt(map.getMapProperty("startX", null));
        startY = Integer.parseInt(map.getMapProperty("startY", null));
        targetX = Integer.parseInt(map.getMapProperty("targetX", null));
        targetY = Integer.parseInt(map.getMapProperty("targetY", null));

        TILESIZE=map.getTileWidth();

        Image normalSheet = new Image("data/sprites/wandering_trader2.png");
        SpriteSheet critterSheet = new SpriteSheet(normalSheet, 32, 64);
        Image[] wanderingNPC = {critterSheet.getSprite(0, 0),critterSheet.getSprite(1, 0),
            critterSheet.getSprite(2, 0), critterSheet.getSprite(3, 0),
            critterSheet.getSprite(4, 0), critterSheet.getSprite(5, 0)};
        wanderingNPCAnim = new Animation(wanderingNPC, 230,true);


        pathmap = new PathMap(map);
        finder = new AStarPathFinder(pathmap, 500, false);
        
        critterAnimation[NORMAL] = spriteLoader.getCritterAnimation(NORMAL);
        critterAnimation[FIRE] = spriteLoader.getCritterAnimation(FIRE);
        critterAnimation[ICE] = spriteLoader.getCritterAnimation(ICE);

        towerFactory = new CudaTowerManager();
        
        critterManager = new CritterManager(startX, startY,
                targetX, targetY, finder, critterAnimation);
        waveNumber = 0;
        critterCount = level.getWave(waveNumber).getNumCritters();
        
        waterAnimation = new renderWater(map.getWidth()+5,map.getHeight());

        Font font = new Font("Verdana", Font.PLAIN, 20);
        trueTypeFont = new TrueTypeFont(font, true);
        startWaves = true;

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


    }

    public void update(GameContainer container, StateBasedGame game, int delta) throws SlickException {

        mouseCounter -= delta;
        waveCounter -= delta;
        generateCounter -= delta;

//        ArrayList<Critter> tempCritterList = new ArrayList<Critter>();

        Input input = container.getInput();

        if (mouseCounter <= 0) {
            
//            if (input.isKeyPressed(Input.KEY_F2)) {
//                game.enterState(TowerDefence.PATHTESTSTATE);
//                mouseCounter=100;
//            }

            if (input.isKeyPressed(Input.KEY_ENTER)) {
                startWaves = true;
                mouseCounter=100;
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
                    int currentXTile = (int) Math.floor((x / CudaTestState.TILESIZE));
                    int currentYTile = (int) Math.floor((y / CudaTestState.TILESIZE));
                    if (button == 0) {
                        if(currentXTile <= 21) {
                            if (pathmap.getTerrain(currentXTile, currentYTile) != PathMap.GRASS
                                    && pathmap.getTerrain(currentXTile, currentYTile) != PathMap.NOPLACE) {
                                try {
                                    towerFactory.addTower(String.valueOf(x),
                                            new Vector2f(currentXTile * CudaTestState.TILESIZE,
                                            currentYTile * CudaTestState.TILESIZE));

                                    pathmap.setTowerTerrain(new Vector2f(currentXTile, currentYTile));

                                    mouseCounter = 100;
                                } catch (SlickException ex) {
                                    Logger.getLogger(CudaTestState.class.getName()).log(Level.SEVERE, null, ex);
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
            }

            public void mouseReleased(int button, int x, int y) {
            }

            public void mouseMoved(int oldx, int oldy, int newx, int newy) {
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

}
